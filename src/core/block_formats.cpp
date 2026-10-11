// The flipbooks' storage formats (include/neuralfx/flipbook.hpp): raw RGBA8, our own BC3 layout, and the production
// block formats BC7 and ASTC through open-source encoders fetched at build time (cmake/encoders.cmake). Without them
// available() is false for those formats and encode() throws.
#include <neuralfx/flipbook.hpp>

#include <array>
#include <cstring>
#include <format>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

#ifndef NEURALFX_HAVE_BC7
#define NEURALFX_HAVE_BC7 0
#endif
#ifndef NEURALFX_HAVE_ASTC
#define NEURALFX_HAVE_ASTC 0
#endif
#if NEURALFX_HAVE_BC7
#include <basisu_bc7e_scalar.h>
#include <bc7decomp.h>
#endif
#if NEURALFX_HAVE_ASTC
#include <astcenc.h>
#endif

namespace nfx::flipbook {

namespace {

constexpr std::array kProduction = {Codec::bc7,     Codec::astc4x4,   Codec::astc5x5,  Codec::astc6x6,
                                    Codec::astc8x8, Codec::astc10x10, Codec::astc12x12};

int blocks_along(int pixels, int block) { return (pixels + block - 1) / block; }

// Bytes of a stored width x height image.
std::size_t stored_bytes(Codec codec, int w, int h) {
  if (codec == Codec::raw) return static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4;
  const int b = block_dim(codec);
  return static_cast<std::size_t>(blocks_along(w, b)) * static_cast<std::size_t>(blocks_along(h, b)) * 16;
}

// Texels of a block-row-major image of 4x4 blocks (row by row, 16 RGBA texels each) to and from an RGBA image.
void gather4x4(std::span<const std::uint8_t> rgba, int w, int bx, int by, std::uint8_t* out) {
  for (int y = 0; y < 4; ++y) {
    std::memcpy(out + y * 16, rgba.data() + (static_cast<std::size_t>(by * 4 + y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(bx) * 4) * 4, 16);
  }
}
void scatter4x4(std::span<std::uint8_t> rgba, int w, int bx, int by, const std::uint8_t* in) {
  for (int y = 0; y < 4; ++y) {
    std::memcpy(rgba.data() + (static_cast<std::size_t>(by * 4 + y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(bx) * 4) * 4, in + y * 16, 16);
  }
}

void need_multiple_of_4(Codec codec, int w, int h) {
  if (w % 4 || h % 4) throw std::invalid_argument(std::format("{}: width and height must be multiples of 4", codec_name(codec)));
}

#if NEURALFX_HAVE_BC7
// BC7 by bc7e at its "veryslow" level (uber level 2, two partitions tried per mode, p-bit search): on the study
// clips' frames it scores as well as its slowest level, 0.2 dB above its basic level.
std::vector<std::uint8_t> encode_bc7(std::span<const std::uint8_t> rgba, int w, int h) {
  static std::once_flag once;
  std::call_once(once, [] { bc7e_scalar::bc7e_compress_block_init(); });
  bc7e_scalar::bc7e_compress_block_params params{};
  bc7e_scalar::bc7e_compress_block_params_init_veryslow(&params, false);  // linear RGBA error, equal weights
  const int bw = w / 4, bh = h / 4;
  const auto n = static_cast<std::size_t>(bw) * static_cast<std::size_t>(bh);
  std::vector<std::uint32_t> texels(n * 16);
  for (int by = 0; by < bh; ++by) {
    for (int bx = 0; bx < bw; ++bx) {
      gather4x4(rgba, w, bx, by, reinterpret_cast<std::uint8_t*>(texels.data() + (static_cast<std::size_t>(by) * static_cast<std::size_t>(bw) + static_cast<std::size_t>(bx)) * 16));
    }
  }
  std::vector<std::uint64_t> blocks(n * 2);
  bc7e_scalar::bc7e_compress_blocks(static_cast<std::uint32_t>(n), blocks.data(), texels.data(), &params);
  std::vector<std::uint8_t> out(n * 16);
  std::memcpy(out.data(), blocks.data(), out.size());  // the blocks' bytes in memory order, as the GPU reads them
  return out;
}

std::vector<std::uint8_t> decode_bc7(std::span<const std::uint8_t> blocks, int w, int h) {
  const int bw = w / 4, bh = h / 4;
  if (blocks.size() != static_cast<std::size_t>(bw) * static_cast<std::size_t>(bh) * 16) throw std::invalid_argument("bc7: wrong number of blocks");
  std::vector<std::uint8_t> out(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
  std::array<bc7decomp::color_rgba, 16> texels;
  for (int by = 0; by < bh; ++by) {
    for (int bx = 0; bx < bw; ++bx) {
      const std::uint8_t* b = blocks.data() + (static_cast<std::size_t>(by) * static_cast<std::size_t>(bw) + static_cast<std::size_t>(bx)) * 16;
      if (!bc7decomp_ref::unpack_bc7(b, texels.data())) throw std::runtime_error("bc7: invalid block");
      scatter4x4(out, w, bx, by, reinterpret_cast<const std::uint8_t*>(texels.data()));
    }
  }
  return out;
}
#endif

#if NEURALFX_HAVE_ASTC
// One astc-encoder context per thread and block size (allocating one builds its tables, a few milliseconds). The
// "thorough" preset: on the study clips' frames "exhaustive" is 0.15 to 0.3 dB better per frame at 4 to 7 times the
// time. The decode_unorm8 flag makes the encoder measure its error after the 8-bit decode the flipbook plays.
struct AstcContext {
  astcenc_context* ctx = nullptr;
  ~AstcContext() { astcenc_context_free(ctx); }
};

astcenc_context* astc_context(int block) {
  thread_local std::map<int, std::unique_ptr<AstcContext>> contexts;
  auto& slot = contexts[block];
  if (!slot) {
    astcenc_config config{};
    const auto b = static_cast<unsigned>(block);
    if (astcenc_config_init(ASTCENC_PRF_LDR, b, b, 1, ASTCENC_PRE_THOROUGH, ASTCENC_FLG_USE_DECODE_UNORM8, &config) != ASTCENC_SUCCESS) {
      throw std::runtime_error(std::format("astc {}x{}: configuration refused", block, block));
    }
    auto c = std::make_unique<AstcContext>();
    if (const astcenc_error e = astcenc_context_alloc(&config, 1, &c->ctx, nullptr); e != ASTCENC_SUCCESS) {
      throw std::runtime_error(std::format("astc {}x{}: {}", block, block, astcenc_get_error_string(e)));
    }
    slot = std::move(c);
  }
  return slot->ctx;
}

astcenc_image astc_image(std::uint8_t* data, int w, int h, void*& slice) {
  slice = data;
  astcenc_image img{};
  img.dim_x = static_cast<unsigned>(w);
  img.dim_y = static_cast<unsigned>(h);
  img.dim_z = 1;
  img.data_type = ASTCENC_TYPE_U8;
  img.data = &slice;
  return img;
}

constexpr astcenc_swizzle kRgba{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};

std::vector<std::uint8_t> encode_astc(int block, std::span<const std::uint8_t> rgba, int w, int h) {
  astcenc_context* ctx = astc_context(block);
  std::vector<std::uint8_t> pixels(rgba.begin(), rgba.end());  // the encoder takes a non-const image
  void* slice = nullptr;
  astcenc_image img = astc_image(pixels.data(), w, h, slice);
  std::vector<std::uint8_t> out(static_cast<std::size_t>(blocks_along(w, block)) * static_cast<std::size_t>(blocks_along(h, block)) * 16);
  const astcenc_error e = astcenc_compress_image(ctx, &img, &kRgba, out.data(), out.size(), 0);
  astcenc_compress_reset(ctx);
  if (e != ASTCENC_SUCCESS) throw std::runtime_error(std::format("astc {}x{}: {}", block, block, astcenc_get_error_string(e)));
  return out;
}

std::vector<std::uint8_t> decode_astc(int block, std::span<const std::uint8_t> blocks, int w, int h) {
  astcenc_context* ctx = astc_context(block);
  std::vector<std::uint8_t> out(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
  void* slice = nullptr;
  astcenc_image img = astc_image(out.data(), w, h, slice);
  const astcenc_error e = astcenc_decompress_image(ctx, blocks.data(), blocks.size(), &img, &kRgba, 0);
  astcenc_decompress_reset(ctx);
  if (e != ASTCENC_SUCCESS) throw std::runtime_error(std::format("astc {}x{}: {}", block, block, astcenc_get_error_string(e)));
  return out;
}
#endif

[[noreturn, maybe_unused]] void unavailable(Codec codec) {
  throw std::runtime_error(std::format("{}: this build has no encoder for it (configure with NEURALFX_FETCH_ENCODERS=ON and network "
                                       "access to github.com)",
                                       codec_name(codec)));
}

}  // namespace

std::string_view codec_name(Codec codec) {
  switch (codec) {
    case Codec::raw: return "raw";
    case Codec::bc3: return "bc3";
    case Codec::bc7: return "bc7";
    case Codec::astc4x4: return "astc4x4";
    case Codec::astc5x5: return "astc5x5";
    case Codec::astc6x6: return "astc6x6";
    case Codec::astc8x8: return "astc8x8";
    case Codec::astc10x10: return "astc10x10";
    case Codec::astc12x12: return "astc12x12";
  }
  return "?";
}

int block_dim(Codec codec) {
  switch (codec) {
    case Codec::raw: return 1;
    case Codec::bc3:
    case Codec::bc7:
    case Codec::astc4x4: return 4;
    case Codec::astc5x5: return 5;
    case Codec::astc6x6: return 6;
    case Codec::astc8x8: return 8;
    case Codec::astc10x10: return 10;
    case Codec::astc12x12: return 12;
  }
  return 1;
}

std::size_t frame_bytes(Codec codec, int res) { return stored_bytes(codec, res, res); }

bool production(Codec codec) { return codec != Codec::raw && codec != Codec::bc3; }

bool available(Codec codec) {
  if (codec == Codec::bc7) return NEURALFX_HAVE_BC7 != 0;
  if (production(codec)) return NEURALFX_HAVE_ASTC != 0;
  return true;
}

std::span<const Codec> production_codecs() { return kProduction; }

std::vector<std::uint8_t> encode(Codec codec, std::span<const std::uint8_t> rgba, int width, int height) {
  if (rgba.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4) throw std::invalid_argument("encode: image size");
  switch (codec) {
    case Codec::raw: return {rgba.begin(), rgba.end()};
    case Codec::bc3: return compress_bc3(rgba, width, height);
    case Codec::bc7:
      need_multiple_of_4(codec, width, height);
#if NEURALFX_HAVE_BC7
      return encode_bc7(rgba, width, height);
#else
      unavailable(codec);
#endif
    default:
#if NEURALFX_HAVE_ASTC
      return encode_astc(block_dim(codec), rgba, width, height);
#else
      unavailable(codec);
#endif
  }
}

std::vector<std::uint8_t> decode(Codec codec, std::span<const std::uint8_t> stored, int width, int height) {
  if (stored.size() != stored_bytes(codec, width, height)) throw std::invalid_argument("decode: stored size");
  switch (codec) {
    case Codec::raw: return {stored.begin(), stored.end()};
    case Codec::bc3: return decompress_bc3(stored, width, height);
    case Codec::bc7:
      need_multiple_of_4(codec, width, height);
#if NEURALFX_HAVE_BC7
      return decode_bc7(stored, width, height);
#else
      unavailable(codec);
#endif
    default:
#if NEURALFX_HAVE_ASTC
      return decode_astc(block_dim(codec), stored, width, height);
#else
      unavailable(codec);
#endif
  }
}

}  // namespace nfx::flipbook
