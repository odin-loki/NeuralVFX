/* nvfx_lit_host: lit sprites for an engine, in plain C (docs/ENGINES.md §11). A stage of the effect library draws its
 * effects into the engine layers (colour, albedo, emission, normal and thickness) beside its picture; this program reads
 * them as an engine would and lights them itself, over a background of its own, with a light that moves.
 *
 *   nvfx_lit_host CATALOGUE_DIR MODELS_DIR OUT_DIR [--frames 120] [--threads 2]
 *       a stage of smoke, a campfire and a rocket trail moved across it, with the layers on; every frame its four
 *       layers are read (RGBA32F) and relit: albedo times (ambient light + a light that circles the scene, by the
 *       normals), plus the emission, over the background by alpha; every 20th frame is written as PNG to OUT_DIR
 *       (lit_NNN.png, and the four layers as RGBA8: colour_, albedo_, emission_, normal_NNN.png)
 *   nvfx_lit_host --self-test CATALOGUE_DIR
 *       procedural effects only (no learned effect needed): the layers are the same for the same calls and on any
 *       number of threads; the picture is the same with the layers on or off; thickness is 0 exactly where alpha is 0,
 *       and the albedo's alpha is the colour's (exit status: the number of failed checks)
 */
#include <neuralfx/nvfx_library.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond)                                      \
  do {                                                   \
    if (!(cond)) {                                       \
      fprintf(stderr, "self-test failed: %s\n", #cond); \
      ++failures;                                        \
    }                                                    \
  } while (0)

/* --- a PNG writer (stored deflate blocks: no compression library needed) ------------------------------------------ */

static unsigned long crc_table[256];
static void crc_init(void) {
  unsigned long c;
  int n, k;
  for (n = 0; n < 256; ++n) {
    c = (unsigned long)n;
    for (k = 0; k < 8; ++k) c = (c & 1) ? 0xedb88320ul ^ (c >> 1) : c >> 1;
    crc_table[n] = c;
  }
}
static unsigned long crc(unsigned long c, const unsigned char* p, size_t n) {
  size_t i;
  for (i = 0; i < n; ++i) c = crc_table[(c ^ p[i]) & 0xff] ^ (c >> 8);
  return c;
}
static void be32(unsigned char* p, unsigned long v) {
  p[0] = (unsigned char)(v >> 24);
  p[1] = (unsigned char)(v >> 16);
  p[2] = (unsigned char)(v >> 8);
  p[3] = (unsigned char)v;
}
static void chunk(FILE* f, const char* type, const unsigned char* data, size_t n) {
  unsigned char b[4];
  unsigned long c = crc(0xfffffffful, (const unsigned char*)type, 4);
  be32(b, (unsigned long)n);
  fwrite(b, 1, 4, f);
  fwrite(type, 1, 4, f);
  if (n) fwrite(data, 1, n, f);
  c = crc(c, data, n) ^ 0xfffffffful;
  be32(b, c);
  fwrite(b, 1, 4, f);
}
static int write_png(const char* path, const unsigned char* rgba, int w, int h) {
  static const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  const size_t row = (size_t)w * 4 + 1, raw = row * (size_t)h, blocks = (raw + 65534) / 65535;
  unsigned char* z = (unsigned char*)malloc(2 + raw + blocks * 5 + 4);
  unsigned char* r = (unsigned char*)malloc(raw);
  unsigned char ihdr[13];
  unsigned long a = 1, b = 0;
  size_t i, o = 0, left;
  int y;
  FILE* f = fopen(path, "wb");
  if (!f || !z || !r) {
    if (f) fclose(f);
    free(z);
    free(r);
    return 0;
  }
  for (y = 0; y < h; ++y) {
    r[(size_t)y * row] = 0;
    memcpy(r + (size_t)y * row + 1, rgba + (size_t)y * (size_t)w * 4, (size_t)w * 4);
  }
  z[o++] = 0x78;
  z[o++] = 0x01;
  for (i = 0, left = raw; left > 0;) {
    const size_t n = left < 65535 ? left : 65535;
    z[o++] = (unsigned char)(left == n ? 1 : 0);
    z[o++] = (unsigned char)(n & 0xff);
    z[o++] = (unsigned char)(n >> 8);
    z[o++] = (unsigned char)(~n & 0xff);
    z[o++] = (unsigned char)((~n >> 8) & 0xff);
    memcpy(z + o, r + i, n);
    o += n;
    i += n;
    left -= n;
  }
  for (i = 0; i < raw; ++i) {
    a = (a + r[i]) % 65521ul;
    b = (b + a) % 65521ul;
  }
  be32(z + o, (b << 16) | a);
  o += 4;
  be32(ihdr, (unsigned long)w);
  be32(ihdr + 4, (unsigned long)h);
  ihdr[8] = 8;  /* 8 bits */
  ihdr[9] = 6;  /* RGBA */
  ihdr[10] = ihdr[11] = ihdr[12] = 0;
  fwrite(sig, 1, 8, f);
  chunk(f, "IHDR", ihdr, 13);
  chunk(f, "IDAT", z, o);
  chunk(f, "IEND", NULL, 0);
  fclose(f);
  free(z);
  free(r);
  return 1;
}

/* --- the stage -------------------------------------------------------------------------------------------------------- */

static nvfx_library* open_library(const char* dir, const char* models) {
  nvfx_library_desc d;
  nvfx_library* lib = NULL;
  nvfx_scene_error err;
  nvfx_library_desc_init(&d);
  d.catalogue_dir = dir;
  d.models_dir = models;
  if (nvfx_library_open(&d, &lib, &err) != NVFX_OK) {
    fprintf(stderr, "library: %s\n", err.message);
    return NULL;
  }
  return lib;
}

static nvfx_stage* make_stage(const nvfx_library* lib, const nvfx_stage_room* rooms, int n, int w, int h, int threads, int layers) {
  nvfx_stage_desc sd;
  nvfx_stage* stage = NULL;
  nvfx_scene_error err;
  nvfx_stage_desc_init(&sd);
  sd.width = w;
  sd.height = h;
  sd.rooms = rooms;
  sd.n_rooms = n;
  sd.threads = threads;
  sd.layers = layers;
  if (nvfx_stage_create(lib, &sd, &stage, &err) != NVFX_OK) {
    fprintf(stderr, "stage: %s\n", err.message);
    return NULL;
  }
  return stage;
}

static unsigned long long fnv(unsigned long long h, const void* p, size_t n) {
  const unsigned char* b = (const unsigned char*)p;
  size_t i;
  for (i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

/* --- relighting, as an engine's shader would ---------------------------------------------------------------------------- */

static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }
static unsigned char to8(float v) { return (unsigned char)(255.f * powf(clampf(v, 0.f, 1.f), 1.f / 2.2f) + 0.5f); }

/* out = albedo * (ambient + light) + emission, over the engine's background by alpha. The light is a point light at
 * (lx, ly) on the screen, lz pixels in front of it, falling off over `range`; the normals' z faces the viewer, their y
 * up (the screen's y goes down). A wrapped diffuse term (n.l + 0.6) / 1.6 lights the thin edges of smoke softly. */
static void relight(const float* albedo, const float* emission, const float* normal, int w, int h, float ground, float lx, float ly, float lz,
                    float range, unsigned char* out) {
  int x, y;
  for (y = 0; y < h; ++y) {
    for (x = 0; x < w; ++x) {
      const size_t i = ((size_t)y * (size_t)w + (size_t)x) * 4;
      const float* A = albedo + i;
      const float* E = emission + i;
      const float* N = normal + i;
      const float fy = (float)y / (float)h;
      float bg[3], dx = lx - (float)x, dy = (float)y - ly, dz = lz, d, nl, k, glow;
      int c;
      if ((float)y < ground) { /* the engine's own sky: dusk, and the light's own disc */
        bg[0] = 0.10f + 0.20f * fy;
        bg[1] = 0.10f + 0.12f * fy;
        bg[2] = 0.22f + 0.05f * fy;
      } else { /* and its ground, lit by the light */
        const float g = 1.f / (1.f + ((lx - (float)x) * (lx - (float)x) + ((float)y - ground) * 9.f) / (range * range));
        bg[0] = 0.05f + 0.25f * g;
        bg[1] = 0.045f + 0.18f * g;
        bg[2] = 0.04f + 0.1f * g;
      }
      glow = 1.f / (1.f + ((float)x - lx) * ((float)x - lx) * 0.02f + ((float)y - ly) * ((float)y - ly) * 0.02f);
      d = sqrtf(dx * dx + dy * dy + dz * dz);
      nl = (N[0] * dx + N[1] * dy + N[2] * dz) / (d + 1e-6f);
      k = (nl + 0.6f) / 1.6f;
      k = k < 0.f ? 0.f : k;
      k = 1.6f * k / (1.f + d * d / (range * range));
      for (c = 0; c < 3; ++c) {
        const float light = c == 0 ? 1.f : c == 1 ? 0.75f : 0.45f;    /* a warm light */
        const float ambient = c == 0 ? 0.10f : c == 1 ? 0.11f : 0.16f; /* and a cool sky */
        const float lit = A[c] * (ambient + light * k) + E[c];
        out[i + (size_t)c] = to8(lit + (bg[c] + 2.f * glow * light) * (1.f - clampf(A[3], 0.f, 1.f)));
      }
      out[i + 3] = 255;
    }
  }
}

static int demo(const char* catalogue, const char* models, const char* out_dir, int frames, int threads) {
  const int W = 640, H = 360;
  const float ground = 313.f;
  nvfx_stage_room rooms[3] = {{"chimney_smoke", 1, 0}, {"campfire", 1, 0}, {"rocket_trail", 1, 0}};
  nvfx_library* lib = open_library(catalogue, models);
  nvfx_stage* stage;
  nvfx_scene* scene;
  float *albedo, *emission, *normal, *colour;
  unsigned char *lit, *rgba;
  nvfx_fx rocket = 0, fx = 0;
  char path[1024];
  int f, written = 0;
  if (!lib) return 1;
  stage = make_stage(lib, rooms, 3, W, H, threads, 1);
  if (!stage) return 1;
  scene = nvfx_stage_scene(stage);
  nvfx_stage_spawn(stage, "chimney_smoke", 430.f, ground, 0.8f, -1.0, &fx);
  nvfx_stage_spawn(stage, "campfire", 200.f, ground, 0.6f, -1.0, &fx);
  nvfx_stage_spawn(stage, "rocket_trail", 40.f, 250.f, 0.8f, -1.0, &rocket);
  colour = (float*)malloc((size_t)W * H * 16);
  albedo = (float*)malloc((size_t)W * H * 16);
  emission = (float*)malloc((size_t)W * H * 16);
  normal = (float*)malloc((size_t)W * H * 16);
  lit = (unsigned char*)malloc((size_t)W * H * 4);
  rgba = (unsigned char*)malloc((size_t)W * H * 4);
  for (f = 1; f <= frames; ++f) {
    const float t = (float)f / 30.f;
    const float lx = 320.f + 260.f * cosf(1.3f * t), ly = 170.f - 110.f * sinf(1.3f * t);
    nvfx_stage_move(stage, rocket, 40.f + 170.f * t, 250.f - 60.f * t + 20.f * sinf(2.f * t));
    nvfx_scene_step_frames(scene, 1);
    /* an engine's frame: the four layers (the picture itself is not needed) */
    nvfx_scene_render_layer(scene, NVFX_LAYER_COLOUR, NVFX_FORMAT_RGBA32F, colour, (size_t)W * 16);
    nvfx_scene_render_layer(scene, NVFX_LAYER_ALBEDO, NVFX_FORMAT_RGBA32F, albedo, (size_t)W * 16);
    nvfx_scene_render_layer(scene, NVFX_LAYER_EMISSION, NVFX_FORMAT_RGBA32F, emission, (size_t)W * 16);
    nvfx_scene_render_layer(scene, NVFX_LAYER_NORMAL, NVFX_FORMAT_RGBA32F, normal, (size_t)W * 16);
    relight(albedo, emission, normal, W, H, ground, lx, ly, 120.f, 260.f, lit);
    if (f % 20 == 0 && out_dir) {
      static const char* names[4] = {"colour", "albedo", "emission", "normal"};
      int l;
      snprintf(path, sizeof path, "%s/lit_%03d.png", out_dir, f);
      written += write_png(path, lit, W, H);
      for (l = 0; l < 4; ++l) {
        nvfx_scene_render_layer(scene, (nvfx_scene_layer)l, NVFX_FORMAT_RGBA8, rgba, (size_t)W * 4);
        snprintf(path, sizeof path, "%s/%s_%03d.png", out_dir, names[l], f);
        written += write_png(path, rgba, W, H);
      }
    }
  }
  printf("lit sprites: %d frames of %dx%d relit by a moving light; %d PNG files in %s\n", frames, W, H, written, out_dir ? out_dir : "(none)");
  free(colour);
  free(albedo);
  free(emission);
  free(normal);
  free(lit);
  free(rgba);
  nvfx_stage_free(stage);
  nvfx_library_free(lib);
  return 0;
}

/* --- the self-test: procedural effects only -------------------------------------------------------------------------- */

typedef struct Run {
  unsigned long long picture, layers[4];
  int covered, uncovered_thick, mismatched_alpha, normals_off;
} Run;

static Run play(const nvfx_library* lib, int threads, int layers) {
  const int W = 320, H = 200, F = 40;
  nvfx_stage_room rooms[3] = {{"sparks", 1, 0}, {"healing_circle", 1, 0}, {"snow", 1, 0}};
  Run r;
  nvfx_stage* stage = make_stage(lib, rooms, 3, W, H, threads, layers);
  nvfx_scene* scene;
  unsigned char* rgba = (unsigned char*)malloc((size_t)W * H * 4);
  float* L[4];
  nvfx_fx fx = 0;
  int f, l;
  size_t i;
  memset(&r, 0, sizeof r);
  if (!stage || !rgba) {
    ++failures;
    free(rgba);
    return r;
  }
  for (l = 0; l < 4; ++l) L[l] = (float*)malloc((size_t)W * H * 16);
  scene = nvfx_stage_scene(stage);
  nvfx_stage_spawn(stage, "sparks", 80.f, 170.f, 1.f, -1.0, &fx);
  nvfx_stage_spawn(stage, "healing_circle", 220.f, 170.f, 0.6f, 0.2, &fx);
  nvfx_stage_spawn(stage, "snow", 160.f, 170.f, 0.4f, -1.0, &fx);
  for (f = 1; f <= F; ++f) {
    if (f == 20) nvfx_stage_move(stage, fx, 150.f, 170.f);
    nvfx_scene_step_frames(scene, 1);
    nvfx_scene_render(scene, rgba, (size_t)W * 4);
  }
  r.picture = fnv(1469598103934665603ull, rgba, (size_t)W * H * 4);
  CHECK(nvfx_scene_render_layer(scene, NVFX_LAYER_COLOUR, NVFX_FORMAT_RGBA32F, L[0], (size_t)W * 16) == (layers ? NVFX_OK : NVFX_ERROR_ARGUMENT));
  if (layers) {
    for (l = 0; l < 4; ++l) {
      CHECK(nvfx_scene_render_layer(scene, (nvfx_scene_layer)l, NVFX_FORMAT_RGBA32F, L[l], (size_t)W * 16) == NVFX_OK);
      r.layers[l] = fnv(1469598103934665603ull, L[l], (size_t)W * H * 16);
    }
    for (i = 0; i < (size_t)W * H; ++i) {
      const float a = L[0][i * 4 + 3], thick = L[3][i * 4 + 3];
      const float n = L[3][i * 4] * L[3][i * 4] + L[3][i * 4 + 1] * L[3][i * 4 + 1] + L[3][i * 4 + 2] * L[3][i * 4 + 2];
      r.covered += a > 0.f;
      r.uncovered_thick += (a == 0.f) != (thick == 0.f);
      r.mismatched_alpha += L[1][i * 4 + 3] != a;
      r.normals_off += fabsf(n - 1.f) > 1e-3f;
    }
  }
  for (l = 0; l < 4; ++l) free(L[l]);
  free(rgba);
  nvfx_stage_free(stage);
  return r;
}

static int self_test(const char* catalogue) {
  nvfx_library* lib = open_library(catalogue, NULL);
  Run a, b, c, d;
  int l;
  if (!lib) return 1;
  a = play(lib, 2, 1);
  b = play(lib, 2, 1);
  c = play(lib, 3, 1);
  d = play(lib, 2, 0);
  for (l = 0; l < 4; ++l) {
    CHECK(a.layers[l] == b.layers[l]); /* the same calls, the same layers */
    CHECK(a.layers[l] == c.layers[l]); /* on any number of threads */
  }
  CHECK(a.picture == b.picture && a.picture == c.picture);
  CHECK(a.picture == d.picture); /* the picture with the layers on is the picture without them */
  CHECK(a.covered > 100);         /* something covers pixels (snow) */
  CHECK(a.uncovered_thick == 0);  /* thickness 0 exactly where alpha is 0 */
  CHECK(a.mismatched_alpha == 0); /* the albedo's alpha is the colour's */
  CHECK(a.normals_off == 0);      /* unit normals everywhere */
  printf("self-test: %d pixels covered, layers %016llx %016llx %016llx %016llx, picture %016llx: %d failed checks\n", a.covered, a.layers[0],
         a.layers[1], a.layers[2], a.layers[3], a.picture, failures);
  nvfx_library_free(lib);
  return failures;
}

int main(int argc, char** argv) {
  int frames = 120, threads = 2, i;
  crc_init();
  if (argc >= 3 && strcmp(argv[1], "--self-test") == 0) return self_test(argv[2]);
  if (argc < 4) {
    fprintf(stderr, "usage: nvfx_lit_host CATALOGUE_DIR MODELS_DIR OUT_DIR [--frames N] [--threads N] | --self-test CATALOGUE_DIR\n");
    return 2;
  }
  for (i = 4; i + 1 < argc; i += 2) {
    if (strcmp(argv[i], "--frames") == 0) frames = atoi(argv[i + 1]);
    else if (strcmp(argv[i], "--threads") == 0) threads = atoi(argv[i + 1]);
  }
  return demo(argv[1], argv[2], argv[3], frames, threads);
}
