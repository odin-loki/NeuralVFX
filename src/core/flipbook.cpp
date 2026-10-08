#include <neuralfx/flipbook.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace nfx::flipbook {

std::string Spec::describe() const {
  std::string s = std::format("{} {}f {}px", codec == Codec::raw ? "raw" : "bc3", frames, res);
  if (flow_res > 0) s += std::format(" +mv{}", flow_res);
  return s;
}

std::size_t memory_bytes(const Spec& spec, int kept) {
  const std::size_t px = static_cast<std::size_t>(spec.res) * spec.res;
  const std::size_t per_frame = spec.codec == Codec::raw ? px * 4 : px;
  const std::size_t flow = spec.flow_res > 0 ? static_cast<std::size_t>(spec.flow_res) * spec.flow_res * 2 : 0;
  return static_cast<std::size_t>(kept) * (per_frame + flow);
}

// --- block compression ------------------------------------------------------------------------------------------

namespace {

struct Rgb {
  float r = 0, g = 0, b = 0;
  Rgb operator+(Rgb o) const { return {r + o.r, g + o.g, b + o.b}; }
  Rgb operator-(Rgb o) const { return {r - o.r, g - o.g, b - o.b}; }
  Rgb operator*(float s) const { return {r * s, g * s, b * s}; }
  float dot(Rgb o) const { return r * o.r + g * o.g + b * o.b; }
};

std::uint16_t to565(Rgb c) {
  const auto q = [](float v, int bits) {
    const int m = (1 << bits) - 1;
    return static_cast<std::uint16_t>(std::clamp(static_cast<int>(std::lround(std::clamp(v, 0.f, 255.f) * static_cast<float>(m) / 255.f)), 0, m));
  };
  return static_cast<std::uint16_t>((q(c.r, 5) << 11) | (q(c.g, 6) << 5) | q(c.b, 5));
}

Rgb from565(std::uint16_t c) {
  const int r = (c >> 11) & 31, g = (c >> 5) & 63, b = c & 31;
  return {static_cast<float>((r << 3) | (r >> 2)), static_cast<float>((g << 2) | (g >> 4)), static_cast<float>((b << 3) | (b >> 2))};
}

std::array<Rgb, 4> palette(std::uint16_t c0, std::uint16_t c1) {
  const Rgb a = from565(c0), b = from565(c1);
  if (c0 > c1) return {a, b, (a * 2.f + b) * (1.f / 3.f), (a + b * 2.f) * (1.f / 3.f)};
  return {a, b, (a + b) * 0.5f, Rgb{}};
}

// Indices for a palette and the squared error they give.
float assign(std::span<const Rgb, 16> px, const std::array<Rgb, 4>& pal, std::array<int, 16>& idx) {
  float err = 0;
  for (int i = 0; i < 16; ++i) {
    float best = std::numeric_limits<float>::max();
    for (int k = 0; k < 4; ++k) {
      const Rgb d = px[static_cast<std::size_t>(i)] - pal[static_cast<std::size_t>(k)];
      const float e = d.dot(d);
      if (e < best) {
        best = e;
        idx[static_cast<std::size_t>(i)] = k;
      }
    }
    err += best;
  }
  return err;
}

}  // namespace

std::array<std::uint8_t, 8> encode_bc1(std::span<const std::uint8_t, 64> rgba) {
  std::array<Rgb, 16> px;
  Rgb mean;
  for (int i = 0; i < 16; ++i) {
    px[static_cast<std::size_t>(i)] = {float(rgba[static_cast<std::size_t>(i * 4)]), float(rgba[static_cast<std::size_t>(i * 4 + 1)]),
                                       float(rgba[static_cast<std::size_t>(i * 4 + 2)])};
    mean = mean + px[static_cast<std::size_t>(i)] * (1.f / 16.f);
  }
  // Principal axis by power iteration on the covariance.
  float cov[3][3] = {};
  for (const Rgb& p : px) {
    const Rgb d = p - mean;
    const float v[3] = {d.r, d.g, d.b};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) cov[a][b] += v[a] * v[b];
    }
  }
  Rgb axis{1.f, 1.f, 1.f};
  for (int it = 0; it < 8; ++it) {
    const Rgb n{cov[0][0] * axis.r + cov[0][1] * axis.g + cov[0][2] * axis.b, cov[1][0] * axis.r + cov[1][1] * axis.g + cov[1][2] * axis.b,
                cov[2][0] * axis.r + cov[2][1] * axis.g + cov[2][2] * axis.b};
    const float len = std::sqrt(n.dot(n));
    if (len < 1e-6f) break;
    axis = n * (1.f / len);
  }
  float lo = std::numeric_limits<float>::max(), hi = -lo;
  for (const Rgb& p : px) {
    const float t = (p - mean).dot(axis);
    lo = std::min(lo, t);
    hi = std::max(hi, t);
  }
  std::uint16_t c0 = to565(mean + axis * hi), c1 = to565(mean + axis * lo);
  if (c0 < c1) std::swap(c0, c1);
  std::array<int, 16> idx{};
  float err = assign(px, palette(c0, c1), idx);
  // Least-squares refinement of the endpoints for the chosen indices (two rounds).
  for (int round = 0; round < 2 && c0 != c1; ++round) {
    constexpr float w[4] = {1.f, 0.f, 2.f / 3.f, 1.f / 3.f};
    float aa = 0, ab = 0, bb = 0;
    Rgb ax, bx;
    for (int i = 0; i < 16; ++i) {
      const float a = w[idx[static_cast<std::size_t>(i)]], b = 1.f - a;
      aa += a * a;
      ab += a * b;
      bb += b * b;
      ax = ax + px[static_cast<std::size_t>(i)] * a;
      bx = bx + px[static_cast<std::size_t>(i)] * b;
    }
    const float det = aa * bb - ab * ab;
    if (std::abs(det) < 1e-6f) break;
    const Rgb e0 = (ax * bb - bx * ab) * (1.f / det), e1 = (bx * aa - ax * ab) * (1.f / det);
    std::uint16_t n0 = to565(e0), n1 = to565(e1);
    if (n0 < n1) std::swap(n0, n1);
    std::array<int, 16> nidx{};
    const float nerr = assign(px, palette(n0, n1), nidx);
    if (nerr >= err) break;
    err = nerr;
    c0 = n0;
    c1 = n1;
    idx = nidx;
  }
  std::uint32_t bits = 0;
  for (int i = 0; i < 16; ++i) bits |= static_cast<std::uint32_t>(idx[static_cast<std::size_t>(i)]) << (2 * i);
  return {static_cast<std::uint8_t>(c0), static_cast<std::uint8_t>(c0 >> 8), static_cast<std::uint8_t>(c1),
          static_cast<std::uint8_t>(c1 >> 8), static_cast<std::uint8_t>(bits), static_cast<std::uint8_t>(bits >> 8),
          static_cast<std::uint8_t>(bits >> 16), static_cast<std::uint8_t>(bits >> 24)};
}

void decode_bc1(std::span<const std::uint8_t, 8> b, std::span<std::uint8_t, 64> rgba) {
  const auto c0 = static_cast<std::uint16_t>(b[0] | (b[1] << 8)), c1 = static_cast<std::uint16_t>(b[2] | (b[3] << 8));
  const std::uint32_t bits = b[4] | (b[5] << 8) | (b[6] << 16) | (static_cast<std::uint32_t>(b[7]) << 24);
  const auto pal = palette(c0, c1);
  for (int i = 0; i < 16; ++i) {
    const Rgb& c = pal[(bits >> (2 * i)) & 3];
    rgba[static_cast<std::size_t>(i * 4)] = static_cast<std::uint8_t>(std::lround(c.r));
    rgba[static_cast<std::size_t>(i * 4 + 1)] = static_cast<std::uint8_t>(std::lround(c.g));
    rgba[static_cast<std::size_t>(i * 4 + 2)] = static_cast<std::uint8_t>(std::lround(c.b));
  }
}

namespace {

std::array<int, 8> bc4_palette(int a0, int a1) {
  std::array<int, 8> p{a0, a1};
  if (a0 > a1) {
    for (int k = 1; k <= 6; ++k) p[static_cast<std::size_t>(k + 1)] = ((7 - k) * a0 + k * a1 + 3) / 7;
  } else {
    for (int k = 1; k <= 4; ++k) p[static_cast<std::size_t>(k + 1)] = ((5 - k) * a0 + k * a1 + 2) / 5;
    p[6] = 0;
    p[7] = 255;
  }
  return p;
}

}  // namespace

std::array<std::uint8_t, 8> encode_bc4(std::span<const std::uint8_t, 16> v) {
  const auto [mn, mx] = std::ranges::minmax(v);
  const int a0 = mx, a1 = mn;
  std::uint64_t bits = 0;
  if (a0 != a1) {
    const auto pal = bc4_palette(a0, a1);
    for (int i = 0; i < 16; ++i) {
      int best = 0, be = 1 << 30;
      for (int k = 0; k < 8; ++k) {
        const int e = std::abs(pal[static_cast<std::size_t>(k)] - v[static_cast<std::size_t>(i)]);
        if (e < be) {
          be = e;
          best = k;
        }
      }
      bits |= static_cast<std::uint64_t>(best) << (3 * i);
    }
  }
  std::array<std::uint8_t, 8> out{static_cast<std::uint8_t>(a0), static_cast<std::uint8_t>(a1)};
  for (int k = 0; k < 6; ++k) out[static_cast<std::size_t>(k + 2)] = static_cast<std::uint8_t>(bits >> (8 * k));
  return out;
}

void decode_bc4(std::span<const std::uint8_t, 8> b, std::span<std::uint8_t, 16> v) {
  std::uint64_t bits = 0;
  for (int k = 0; k < 6; ++k) bits |= static_cast<std::uint64_t>(b[static_cast<std::size_t>(k + 2)]) << (8 * k);
  const auto pal = bc4_palette(b[0], b[1]);
  for (int i = 0; i < 16; ++i) v[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(pal[(bits >> (3 * i)) & 7]);
}

std::vector<std::uint8_t> compress_bc3(std::span<const std::uint8_t> rgba, int w, int h) {
  if (w % 4 || h % 4) throw std::invalid_argument("bc3: size must be a multiple of 4");
  std::vector<std::uint8_t> out;
  out.reserve(static_cast<std::size_t>(w) * h);
  std::array<std::uint8_t, 64> block{};
  std::array<std::uint8_t, 16> alpha{};
  for (int by = 0; by < h; by += 4) {
    for (int bx = 0; bx < w; bx += 4) {
      for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
          const std::size_t s = (static_cast<std::size_t>(by + y) * w + bx + x) * 4, d = static_cast<std::size_t>(y * 4 + x);
          std::copy_n(rgba.begin() + static_cast<std::ptrdiff_t>(s), 4, block.begin() + static_cast<std::ptrdiff_t>(d * 4));
          alpha[d] = rgba[s + 3];
        }
      }
      const auto a = encode_bc4(alpha);
      const auto c = encode_bc1(block);
      out.insert(out.end(), a.begin(), a.end());
      out.insert(out.end(), c.begin(), c.end());
    }
  }
  return out;
}

std::vector<std::uint8_t> decompress_bc3(std::span<const std::uint8_t> blocks, int w, int h) {
  std::vector<std::uint8_t> out(static_cast<std::size_t>(w) * h * 4);
  std::array<std::uint8_t, 64> block{};
  std::array<std::uint8_t, 16> alpha{};
  std::size_t at = 0;
  for (int by = 0; by < h; by += 4) {
    for (int bx = 0; bx < w; bx += 4) {
      decode_bc4(blocks.subspan(at).first<8>(), alpha);
      decode_bc1(blocks.subspan(at + 8).first<8>(), block);
      at += 16;
      for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
          const std::size_t d = (static_cast<std::size_t>(by + y) * w + bx + x) * 4, s = static_cast<std::size_t>(y * 4 + x);
          std::copy_n(block.begin() + static_cast<std::ptrdiff_t>(s * 4), 3, out.begin() + static_cast<std::ptrdiff_t>(d));
          out[d + 3] = alpha[s];
        }
      }
    }
  }
  return out;
}

// --- flipbooks ---------------------------------------------------------------------------------------------------

std::vector<std::uint8_t> downsample(std::span<const std::uint8_t> rgba, int size, int res) {
  if (res == size) return {rgba.begin(), rgba.end()};
  if (size % res) throw std::invalid_argument("downsample: size must be a multiple of res");
  const int f = size / res;
  std::vector<std::uint8_t> out(static_cast<std::size_t>(res) * res * 4);
  for (int y = 0; y < res; ++y) {
    for (int x = 0; x < res; ++x) {
      for (int c = 0; c < 4; ++c) {
        int s = 0;
        for (int j = 0; j < f; ++j) {
          for (int i = 0; i < f; ++i) s += rgba[(static_cast<std::size_t>(y * f + j) * size + x * f + i) * 4 + static_cast<std::size_t>(c)];
        }
        out[(static_cast<std::size_t>(y) * res + x) * 4 + static_cast<std::size_t>(c)] = static_cast<std::uint8_t>((s + f * f / 2) / (f * f));
      }
    }
  }
  return out;
}

void sample_bilinear(std::span<const std::uint8_t> rgba, int res, float x, float y, std::span<float, 4> out) {
  x = std::clamp(x, 0.f, static_cast<float>(res - 1));
  y = std::clamp(y, 0.f, static_cast<float>(res - 1));
  const int x0 = std::min(static_cast<int>(x), res - 2 < 0 ? 0 : res - 2), y0 = std::min(static_cast<int>(y), res - 2 < 0 ? 0 : res - 2);
  const int x1 = std::min(x0 + 1, res - 1), y1 = std::min(y0 + 1, res - 1);
  const float fx = x - static_cast<float>(x0), fy = y - static_cast<float>(y0);
  for (std::size_t c = 0; c < 4; ++c) {
    const auto at = [&](int xx, int yy) { return static_cast<float>(rgba[(static_cast<std::size_t>(yy) * res + xx) * 4 + c]); };
    const float a = at(x0, y0) + fx * (at(x1, y0) - at(x0, y0));
    const float b = at(x0, y1) + fx * (at(x1, y1) - at(x0, y1));
    out[c] = a + fy * (b - a);
  }
}

namespace {

// Flow from image a to image b (both size x size RGBA), on a flow_res grid, by block matching on luminance + alpha
// with a +-range search, then a 3x3 smoothing. Vectors in output pixels: a(x) ~ b(x + v).
std::vector<float> estimate_flow(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b, int size, int flow_res) {
  const int cell = size / flow_res, half = std::max(4, cell);
  const int range = std::max(4, size / 4);  // up to a quarter of the sprite between kept frames
  const auto val = [&](std::span<const std::uint8_t> im, int x, int y) {
    x = std::clamp(x, 0, size - 1);
    y = std::clamp(y, 0, size - 1);
    const std::size_t i = (static_cast<std::size_t>(y) * size + x) * 4;
    return static_cast<int>(im[i]) + im[i + 1] + im[i + 2] + 2 * im[i + 3];
  };
  std::vector<float> flow(static_cast<std::size_t>(flow_res) * flow_res * 2, 0.f);
  for (int fy = 0; fy < flow_res; ++fy) {
    for (int fx = 0; fx < flow_res; ++fx) {
      const int cx = fx * cell + cell / 2, cy = fy * cell + cell / 2;
      long best = std::numeric_limits<long>::max();
      int bdx = 0, bdy = 0;
      long energy = 0;
      for (int j = -half; j < half; j += 2) {
        for (int i = -half; i < half; i += 2) energy += val(a, cx + i, cy + j);
      }
      if (energy == 0) continue;  // empty block: no motion
      const auto try_at = [&](int dx, int dy) {
        long sad = 4L * (dx * dx + dy * dy);  // prefer small motion on ties
        for (int j = -half; j < half && sad < best; j += 2) {
          for (int i = -half; i < half; i += 2) sad += std::abs(val(a, cx + i, cy + j) - val(b, cx + i + dx, cy + j + dy));
        }
        if (sad < best) {
          best = sad;
          bdx = dx;
          bdy = dy;
        }
      };
      // Coarse search on even displacements, then the 3x3 neighbourhood of the best.
      for (int dy = -range; dy <= range; dy += 2) {
        for (int dx = -range; dx <= range; dx += 2) try_at(dx, dy);
      }
      const int cxb = bdx, cyb = bdy;
      for (int dy = cyb - 1; dy <= cyb + 1; ++dy) {
        for (int dx = cxb - 1; dx <= cxb + 1; ++dx) try_at(dx, dy);
      }
      flow[(static_cast<std::size_t>(fy) * flow_res + fx) * 2] = static_cast<float>(bdx);
      flow[(static_cast<std::size_t>(fy) * flow_res + fx) * 2 + 1] = static_cast<float>(bdy);
    }
  }
  std::vector<float> smooth(flow.size());
  for (int fy = 0; fy < flow_res; ++fy) {
    for (int fx = 0; fx < flow_res; ++fx) {
      for (int c = 0; c < 2; ++c) {
        float s = 0;
        int n = 0;
        for (int j = -1; j <= 1; ++j) {
          for (int i = -1; i <= 1; ++i) {
            const int x = fx + i, y = fy + j;
            if (x < 0 || y < 0 || x >= flow_res || y >= flow_res) continue;
            s += flow[(static_cast<std::size_t>(y) * flow_res + x) * 2 + static_cast<std::size_t>(c)];
            ++n;
          }
        }
        // Stored as 8-bit fixed point (1/4 pixel steps, +-32 pixels): the precision a shipped flipbook would keep.
        const float v = std::clamp(std::round(s / static_cast<float>(n) * 4.f) / 4.f, -32.f, 31.75f);
        smooth[(static_cast<std::size_t>(fy) * flow_res + fx) * 2 + static_cast<std::size_t>(c)] = v;
      }
    }
  }
  return smooth;
}

}  // namespace

Flipbook build(const Clip& ref, const Spec& spec, std::span<const int> keep) {
  if (spec.res <= 0 || ref.size % spec.res) throw std::invalid_argument("flipbook: res must divide the clip size");
  Flipbook fb;
  fb.spec = spec;
  fb.out_size = ref.size;
  fb.source_frames = ref.frames;
  fb.loop = ref.loop;
  if (keep.empty()) {
    const int k = std::clamp(spec.frames, 1, ref.frames);
    for (int i = 0; i < k; ++i) {
      fb.kept.push_back(static_cast<int>(std::lround(static_cast<double>(i) * ref.frames / k)) % ref.frames);
    }
    if (!ref.loop && k > 1) {  // a one-shot clip keeps its last frame too
      for (int i = 0; i < k; ++i) fb.kept[static_cast<std::size_t>(i)] = static_cast<int>(std::lround(static_cast<double>(i) * (ref.frames - 1) / (k - 1)));
    }
  } else {
    fb.kept.assign(keep.begin(), keep.end());
  }
  std::ranges::sort(fb.kept);
  fb.kept.erase(std::ranges::unique(fb.kept).begin(), fb.kept.end());
  fb.spec.frames = static_cast<int>(fb.kept.size());
  for (const int f : fb.kept) {
    auto small = downsample(ref.frame(f), ref.size, spec.res);
    if (spec.codec == Codec::bc3) small = decompress_bc3(compress_bc3(small, spec.res, spec.res), spec.res, spec.res);
    fb.frames.push_back(std::move(small));
  }
  if (spec.flow_res > 0) {
    const std::size_t k = fb.kept.size();
    for (std::size_t i = 0; i < k; ++i) {
      const bool last = i + 1 == k;
      if (last && !ref.loop) {
        fb.flow.emplace_back(static_cast<std::size_t>(spec.flow_res) * spec.flow_res * 2, 0.f);
        continue;
      }
      // Estimated on the reference frames at full resolution (an authoring tool has them).
      fb.flow.push_back(estimate_flow(ref.frame(fb.kept[i]), ref.frame(fb.kept[last ? 0 : i + 1]), ref.size, spec.flow_res));
    }
  }
  fb.bytes = memory_bytes(fb.spec, static_cast<int>(fb.kept.size()));
  return fb;
}

Clip play(const Flipbook& fb) {
  Clip out;
  out.allocate(fb.out_size, fb.source_frames);
  out.loop = fb.loop;
  const int k = static_cast<int>(fb.kept.size());
  const int S = fb.out_size, R = fb.spec.res;
  const float scale = static_cast<float>(R) / static_cast<float>(S);
  std::array<float, 4> a{}, b{};
  for (int f = 0; f < fb.source_frames; ++f) {
    // The kept frames that bracket time f, and the blend weight between them.
    int i0 = k - 1;
    for (int i = 0; i < k; ++i) {
      if (fb.kept[static_cast<std::size_t>(i)] <= f) i0 = i;
    }
    int i1 = i0 + 1;
    float t0 = static_cast<float>(fb.kept[static_cast<std::size_t>(i0)]), t1;
    if (i1 < k) {
      t1 = static_cast<float>(fb.kept[static_cast<std::size_t>(i1)]);
    } else if (fb.loop) {
      i1 = 0;
      t1 = static_cast<float>(fb.kept[0] + fb.source_frames);
    } else {
      i1 = i0;
      t1 = t0 + 1.f;
    }
    float ff = static_cast<float>(f);
    if (ff < t0) ff += static_cast<float>(fb.source_frames);  // before the first kept frame of a loop
    const float w = std::clamp((ff - t0) / std::max(1.f, t1 - t0), 0.f, 1.f);
    const auto& A = fb.frames[static_cast<std::size_t>(i0)];
    const auto& B = fb.frames[static_cast<std::size_t>(i1)];
    const bool mv = fb.spec.flow_res > 0 && i1 != i0;
    auto frame = out.frame(f);
    for (int y = 0; y < S; ++y) {
      for (int x = 0; x < S; ++x) {
        float vx = 0, vy = 0;
        if (mv) {
          std::array<float, 4> fl{};
          const auto& F = fb.flow[static_cast<std::size_t>(i0)];
          const int fr = fb.spec.flow_res;
          const float cell = static_cast<float>(S) / static_cast<float>(fr);
          const float gx = std::clamp((static_cast<float>(x) + 0.5f) / cell - 0.5f, 0.f, static_cast<float>(fr - 1));
          const float gy = std::clamp((static_cast<float>(y) + 0.5f) / cell - 0.5f, 0.f, static_cast<float>(fr - 1));
          const int x0 = std::min(static_cast<int>(gx), fr - 1), y0 = std::min(static_cast<int>(gy), fr - 1);
          const int x1 = std::min(x0 + 1, fr - 1), y1 = std::min(y0 + 1, fr - 1);
          const float ax = gx - static_cast<float>(x0), ay = gy - static_cast<float>(y0);
          for (int c = 0; c < 2; ++c) {
            const auto at = [&](int xx, int yy) { return F[(static_cast<std::size_t>(yy) * fr + xx) * 2 + static_cast<std::size_t>(c)]; };
            const float p = at(x0, y0) + ax * (at(x1, y0) - at(x0, y0)), q = at(x0, y1) + ax * (at(x1, y1) - at(x0, y1));
            fl[static_cast<std::size_t>(c)] = p + ay * (q - p);
          }
          vx = fl[0];
          vy = fl[1];
        }
        // Frame i0 pulled forward by w of the motion, frame i1 pulled back by (1 - w).
        const float sx = (static_cast<float>(x) + 0.5f) * scale - 0.5f, sy = (static_cast<float>(y) + 0.5f) * scale - 0.5f;
        sample_bilinear(A, R, sx - w * vx * scale, sy - w * vy * scale, a);
        sample_bilinear(B, R, sx + (1.f - w) * vx * scale, sy + (1.f - w) * vy * scale, b);
        for (std::size_t c = 0; c < 4; ++c) {
          frame[(static_cast<std::size_t>(y) * S + x) * 4 + c] =
              static_cast<std::uint8_t>(std::clamp(a[c] + w * (b[c] - a[c]), 0.f, 255.f) + 0.5f);
        }
      }
    }
  }
  return out;
}

std::vector<Spec> ladder(int size, int frames) {
  std::vector<Spec> v;
  for (const int res : {size, size / 2, size / 4}) {
    for (int k = frames; k >= 4; k /= 2) v.push_back({k, res, Codec::bc3, 0});
  }
  for (const int res : {size / 2, size / 4}) {
    for (int k = frames; k >= 4; k /= 2) v.push_back({k, res, Codec::raw, 0});
  }
  for (const int res : {size, size / 2}) {
    for (int k = frames / 4; k >= 4; k /= 2) v.push_back({k, res, Codec::bc3, std::max(4, res / 4)});
  }
  return v;
}

}  // namespace nfx::flipbook
