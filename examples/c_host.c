/* nvfx_c_host: how an engine uses the runtime, in plain C (docs/ENGINES.md).
 *
 *   nvfx_c_host effect.nvfx [size] [seconds] [frame.pam]   play the effect for `seconds` at 30 Hz, print the cost per
 *                                                         frame, and optionally write the last frame
 *   nvfx_c_host --scene script.nvfxs [--effects DIR] [--threads 2] [--no-overlap] [--frames N] [--out frame.pam]
 *               [--expect profile.csv]
 *                                                play a scene script (nvfx_scene.h) one frame at a time: print the
 *                                                cost per frame and the heat on screen, write the last frame, and
 *                                                compare every frame's RGB checksum with the rgb_fnv column of a
 *                                                profile written by nvfx_scene_script --profile (exit 1 when one
 *                                                differs; 77, skipped, when the effects or the profile are missing)
 *   nvfx_c_host --self-test                      check the error paths of the API (no model needed)
 */
#include <neuralfx/nvfx.h>
#include <neuralfx/nvfx_scene.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static int self_test(void) {
  int failures = 0;
  nvfx_effect* effect = NULL;
  const char junk[] = "this is not an effect";
#define CHECK(cond)                                         \
  do {                                                      \
    if (!(cond)) {                                          \
      fprintf(stderr, "self-test failed: %s\n", #cond);    \
      ++failures;                                           \
    }                                                       \
  } while (0)
  CHECK(nvfx_effect_load(NULL, &effect) == NVFX_ERROR_ARGUMENT);
  CHECK(nvfx_effect_load("/nonexistent/effect.nvfx", &effect) == NVFX_ERROR_IO);
  CHECK(nvfx_effect_load_memory(junk, sizeof junk, &effect) == NVFX_ERROR_FORMAT);
  CHECK(nvfx_instance_create(NULL, 128, NULL) == NVFX_ERROR_ARGUMENT);
  CHECK(nvfx_render(NULL, 0.0, NULL, 0) == NVFX_ERROR_ARGUMENT);
  CHECK(strcmp(nvfx_status_string(NVFX_OK), "ok") == 0);
  CHECK(nvfx_get_isa() != NVFX_ISA_AUTO);
  {
    /* scenes: script errors carry their line and column; a missing effect names its `effect` statement */
    nvfx_scene_error err;
    nvfx_scene_desc d;
    nvfx_scene* scene = NULL;
    CHECK(nvfx_scene_check("scene size 320 x 180\nfield f = wind, velocity (1, 0), strenght 3", "t", &err) == NVFX_ERROR_SCRIPT);
    CHECK(err.line == 2 && err.column == 34);
    CHECK(strncmp(err.message, "t:2:34: 'strenght' is not a property of field", 45) == 0);
    CHECK(nvfx_scene_check("input breeze = 0.5\neffect e = \"e.nvfx\"\nmodule m = e, size 32, wind breeze", "t", &err) == NVFX_OK);
    CHECK(err.line == 0 && err.message[0] == '\0');
    CHECK(nvfx_scene_check("scene size 64 x 64\nscene length 2", "t", &err) == NVFX_ERROR_SCRIPT && err.line == 2 && err.column == 1);
    nvfx_scene_desc_init(&d);
    CHECK(d.threads == 2 && d.overlap == -1);
    d.script = "scene size 64 x 64\n\neffect fire = \"fire.nvfx\"\nmodule m = fire, size 32";
    d.effects_dir = "/nonexistent";
    CHECK(nvfx_scene_create(&d, &scene, &err) == NVFX_ERROR_IO && scene == NULL);
    CHECK(err.line == 3 && err.column == 1 && strstr(err.message, "cannot load effect 'fire'") != NULL);
    d.script = NULL;
    d.script_path = "/nonexistent/scene.nvfxs";
    CHECK(nvfx_scene_create(&d, &scene, &err) == NVFX_ERROR_IO && err.line == 0);
    CHECK(nvfx_scene_create(NULL, &scene, &err) == NVFX_ERROR_ARGUMENT);
    CHECK(nvfx_scene_render(NULL, NULL, 0) == NVFX_ERROR_ARGUMENT);
    CHECK(nvfx_scene_step(NULL, 0.1, NULL) == NVFX_ERROR_ARGUMENT);
    CHECK(strcmp(nvfx_status_string(NVFX_ERROR_SCRIPT), "error in the scene script") == 0);
    nvfx_scene_free(NULL);
  }
  printf("self-test: %s (ISA %d)\n", failures ? "FAILED" : "ok", (int)nvfx_get_isa());
  return failures ? 1 : 0;
#undef CHECK
}

static void write_pam_rect(const char* path, const uint8_t* rgba, int w, int h) {
  FILE* f = fopen(path, "wb");
  if (!f) return;
  fprintf(f, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", w, h);
  fwrite(rgba, 1, (size_t)w * (size_t)h * 4, f);
  fclose(f);
}

static void write_pam(const char* path, const uint8_t* rgba, int size) { write_pam_rect(path, rgba, size, size); }

/* FNV-1a, 64 bits, over the RGB of every pixel in order: nvfx_scene_script's rgb_fnv. */
static uint64_t rgb_fnv(const uint8_t* rgba, int w, int h, size_t stride) {
  uint64_t hash = 0xcbf29ce484222325ull;
  for (int y = 0; y < h; ++y) {
    const uint8_t* p = rgba + (size_t)y * stride;
    for (int x = 0; x < w; ++x, p += 4) {
      for (int c = 0; c < 3; ++c) hash = (hash ^ p[c]) * 0x100000001b3ull;
    }
  }
  return hash;
}

/* The rgb_fnv column (the last) of nvfx_scene_script's profile, by frame; returns the number of frames read. */
static int read_profile(const char* path, uint64_t* sums, int max_frames) {
  FILE* f = fopen(path, "r");
  char line[4096];
  int n = 0;
  if (!f) return -1;
  while (fgets(line, sizeof line, f)) {
    const char* last = strrchr(line, ',');
    int frame;
    if (!last || sscanf(line, "%d,", &frame) != 1 || frame < 0 || frame >= max_frames) continue;  /* the header */
    sums[frame] = strtoull(last + 1, NULL, 16);
    if (frame + 1 > n) n = frame + 1;
  }
  fclose(f);
  return n;
}

static int play_scene(int argc, char** argv) {
  nvfx_scene_desc d;
  nvfx_scene_error err;
  nvfx_scene_info info;
  nvfx_scene* scene = NULL;
  const char *out = NULL, *expect = NULL;
  int frames = -1, differ = 0, compared = 0, have = 0;
  double total = 0, worst = 0, t_create, t_copy, t_restart;
  uint64_t* want = NULL;
  uint8_t* rgba;
  nvfx_status s;
  nvfx_scene_desc_init(&d);
  d.script_path = argv[2];
  for (int i = 3; i < argc; ++i) {
    const int more = i + 1 < argc;
    if (strcmp(argv[i], "--effects") == 0 && more) d.effects_dir = argv[++i];
    else if (strcmp(argv[i], "--threads") == 0 && more) d.threads = atoi(argv[++i]);
    else if (strcmp(argv[i], "--frames") == 0 && more) frames = atoi(argv[++i]);
    else if (strcmp(argv[i], "--out") == 0 && more) out = argv[++i];
    else if (strcmp(argv[i], "--expect") == 0 && more) expect = argv[++i];
    else if (strcmp(argv[i], "--no-overlap") == 0) d.overlap = 0;
    else {
      fprintf(stderr, "nvfx_c_host --scene: unknown option %s\n", argv[i]);
      return 2;
    }
  }
  t_create = now_ms();
  s = nvfx_scene_create(&d, &scene, &err);
  t_create = now_ms() - t_create;
  if (s != NVFX_OK) {
    fprintf(stderr, "scene: %s (%s)\n", err.message, nvfx_status_string(s));
    return s == NVFX_ERROR_IO && expect ? 77 : 1;
  }
  nvfx_scene_get_info(scene, &info);
  if (frames < 0 || frames > info.frames) frames = info.frames;
  printf("%s: %dx%d, %d frames at %.0f fps, %d modules, %d rules, %d inputs, %d threads%s, %.1f MB of module memory; created in %.0f ms\n", argv[2],
         info.width, info.height, frames, info.fps, info.n_modules, info.n_rules, info.n_inputs, info.threads, info.overlap ? " (overlapped)" : "",
         (double)info.scratch_bytes / 1048576.0, t_create);
  if (expect) {
    want = (uint64_t*)calloc((size_t)frames + 1, sizeof *want);
    have = want ? read_profile(expect, want, frames) : -1;
    if (have <= 0) {
      fprintf(stderr, "cannot read %s: skipped\n", expect);
      free(want);
      nvfx_scene_free(scene);
      return 77;
    }
  }
  rgba = (uint8_t*)malloc((size_t)info.width * (size_t)info.height * 4);
  for (int f = 0; f < frames; ++f) { /* the engine's update: one step and one picture per scene frame */
    const double t0 = now_ms();
    float hot = 0.f, mean = 0.f;
    if (f > 0) nvfx_scene_step_frames(scene, 1);
    nvfx_scene_render(scene, rgba, (size_t)info.width * 4);
    const double dt = now_ms() - t0;
    total += dt;
    if (f > 0 && dt > worst) worst = dt;
    if (want && f < have) {
      const uint64_t got = rgb_fnv(rgba, info.width, info.height, (size_t)info.width * 4);
      ++compared;
      if (got != want[f]) {
        if (differ < 5) fprintf(stderr, "frame %d: rgb_fnv %016llx, the script runner's %016llx\n", f, (unsigned long long)got, (unsigned long long)want[f]);
        ++differ;
      }
    }
    if (f % 30 == 0) {
      nvfx_scene_field_region(scene, NVFX_FIELD_HEAT, 0.f, 0.f, (float)info.scene_width, info.ground, &hot, &mean);
      printf("  t %4.1f s: %6.1f ms, the hottest heat on screen %.2f\n", nvfx_scene_time(scene), dt, hot);
    }
  }
  printf("%d frames: %.2f ms mean, %.2f ms worst\n", frames, total / (frames > 0 ? frames : 1), worst);
  t_copy = now_ms();
  nvfx_scene_render(scene, rgba, (size_t)info.width * 4); /* drawn already: a copy into RGBA */
  t_copy = now_ms() - t_copy;
  t_restart = now_ms();
  nvfx_scene_restart(scene);
  t_restart = now_ms() - t_restart;
  printf("a drawn frame copied again: %.2f ms; a restart (the scene rebuilt): %.0f ms\n", t_copy, t_restart);
  if (want) printf("checksums: %d of %d frames the same as %s\n", compared - differ, compared, expect);
  if (out) write_pam_rect(out, rgba, info.width, info.height);
  free(rgba);
  free(want);
  nvfx_scene_free(scene);
  return differ > 0 || (want && compared == 0) ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && strcmp(argv[1], "--self-test") == 0) return self_test();
  if (argc >= 3 && strcmp(argv[1], "--scene") == 0) return play_scene(argc, argv);
  if (argc < 2) {
    fprintf(stderr, "usage: nvfx_c_host effect.nvfx [size] [seconds] [frame.pam] | --self-test\n");
    return 2;
  }
  nvfx_effect* effect = NULL;
  nvfx_status s = nvfx_effect_load(argv[1], &effect);
  if (s != NVFX_OK) {
    fprintf(stderr, "load failed: %s\n", nvfx_status_string(s));
    return 1;
  }
  nvfx_effect_info info;
  nvfx_effect_get_info(effect, &info);
  const int size = argc >= 3 ? atoi(argv[2]) : info.native_size;
  const double seconds = argc >= 4 ? atof(argv[3]) : 4.0;
  printf("%s: %s, native %d px, %d frames at %.0f fps, %d controls, %d variations, %.1f KB stored, %.1f KB resident\n",
         argv[1], info.loops ? "looping" : "one-shot", info.native_size, info.frames, info.fps, info.n_controls,
         info.n_variations, info.stored_bytes / 1024.0, info.resident_bytes / 1024.0);

  nvfx_instance* inst = NULL;
  s = nvfx_instance_create(effect, size, &inst);
  if (s != NVFX_OK) {
    fprintf(stderr, "instance failed: %s\n", nvfx_status_string(s));
    nvfx_effect_free(effect);
    return 1;
  }
  float controls[8] = {0.8f, 0.3f, 0.6f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};  /* e.g. intensity, wind, turbulence */
  nvfx_instance_set_controls(inst, controls, info.n_controls);
  nvfx_instance_set_seed(inst, 12345);
  uint8_t* rgba = (uint8_t*)malloc((size_t)size * (size_t)size * 4);
  const int frames = (int)(seconds * 30.0);
  double total = 0, worst = 0;
  for (int f = 0; f < frames; ++f) { /* the engine's update: one evaluation per effect frame */
    const double t0 = now_ms();
    nvfx_render(inst, f / 30.0, rgba, (size_t)size * 4);
    const double dt = now_ms() - t0;
    total += dt;
    if (f > 0 && dt > worst) worst = dt;
  }
  printf("%d frames at %dx%d: %.3f ms mean, %.3f ms worst, %.0f MAC/px, %zu bytes scratch\n", frames, size, size,
         total / frames, worst, nvfx_instance_macs_per_pixel(inst), nvfx_instance_scratch_bytes(inst));
  if (argc >= 5) write_pam(argv[4], rgba, size);
  free(rgba);
  nvfx_instance_free(inst);
  nvfx_effect_free(effect);
  return 0;
}
