/* nvfx_c_host: how an engine uses the runtime, in plain C (docs/ENGINES.md).
 *
 *   nvfx_c_host effect.nvfx [size] [seconds] [frame.pam]   play the effect for `seconds` at 30 Hz, print the cost per
 *                                                         frame, and optionally write the last frame
 *   nvfx_c_host --self-test                      check the error paths of the API (no model needed)
 */
#include <neuralfx/nvfx.h>

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
  printf("self-test: %s (ISA %d)\n", failures ? "FAILED" : "ok", (int)nvfx_get_isa());
  return failures ? 1 : 0;
#undef CHECK
}

static void write_pam(const char* path, const uint8_t* rgba, int size) {
  FILE* f = fopen(path, "wb");
  if (!f) return;
  fprintf(f, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", size, size);
  fwrite(rgba, 1, (size_t)size * (size_t)size * 4, f);
  fclose(f);
}

int main(int argc, char** argv) {
  if (argc >= 2 && strcmp(argv[1], "--self-test") == 0) return self_test();
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
