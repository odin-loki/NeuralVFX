/* The volume C API (nvfx_volume.h; study N, docs/STUDY_3D.md) from plain C, as an engine uses it: load a 3D effect, play
 * it, read its extinction as a half-float 3D texture each frame (what a GPU volume renderer would upload), and draw it
 * for a camera that circles it, writing the last picture as a PPM over a dark background.
 *
 *   nvfx_volume_host EFFECT.nvfx3 OUT.ppm [seconds] [threads]
 *   nvfx_volume_host --self-test        the error paths and the camera helper, without an effect file
 */
#include <neuralfx/nvfx_volume.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double seconds_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int self_test(void) {
  int failures = 0;
  uint32_t features = 0;
  nvfx_volume_effect* e = NULL;
  nvfx_volume* v = NULL;
  nvfx_volume_camera cam;
  nvfx_volume_look_at la = {{1.f, 1.f, 1.f}, {0.f, 0.5f, 0.f}, {0.f, 1.f, 0.f}, 45.f, 1.f, 0.f, 0.f};
  const char junk[16] = "NVFXVOL1junk....";
  nvfx_version(&features);
  failures += (features & NVFX_FEATURE_VOLUME) == 0;
  failures += nvfx_volume_effect_load("/nonexistent/fire3d.nvfx3", &e) != NVFX_ERROR_IO;
  failures += nvfx_volume_effect_load_memory(junk, sizeof(junk), &e) != NVFX_ERROR_FORMAT;
  failures += nvfx_volume_effect_load(NULL, &e) != NVFX_ERROR_ARGUMENT;
  failures += nvfx_volume_create(NULL, 1, &v) != NVFX_ERROR_ARGUMENT;
  failures += nvfx_volume_step_to(NULL, 0.0) != NVFX_ERROR_ARGUMENT;
  failures += nvfx_volume_camera_look_at(&la, &cam) != NVFX_OK;
  failures += cam.proj[11] != -1.f; /* a perspective projection */
  la.fov_y_degrees = 0.f;
  failures += nvfx_volume_camera_look_at(&la, &cam) != NVFX_ERROR_ARGUMENT;
  failures += nvfx_volume_effect_control_name(NULL, 0) != NULL;
  printf("volume host self-test: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}

int main(int argc, char** argv) {
  nvfx_volume_effect* e = NULL;
  nvfx_volume* v = NULL;
  nvfx_volume_info info;
  const int w = 480, h = 360;
  double seconds, step_time = 0.0, read_time = 0.0, draw_time = 0.0;
  int threads, frames, f;
  size_t texels;
  uint16_t* tex;
  uint8_t* rgba;
  FILE* out;
  if (argc >= 2 && strcmp(argv[1], "--self-test") == 0) return self_test();
  if (argc < 3) {
    fprintf(stderr, "usage: nvfx_volume_host EFFECT.nvfx3 OUT.ppm [seconds] [threads] | --self-test\n");
    return 2;
  }
  seconds = argc > 3 ? atof(argv[3]) : 3.0;
  threads = argc > 4 ? atoi(argv[4]) : 2;
  if (nvfx_volume_effect_load(argv[1], &e) != NVFX_OK || nvfx_volume_create(e, threads, &v) != NVFX_OK) {
    fprintf(stderr, "cannot load %s\n", argv[1]);
    return 1;
  }
  nvfx_volume_effect_get_info(e, &info);
  printf("%s: %d x %d x %d learned state, %d x %d x %d fields, %d start points, %.1f KB stored, %.1f MB per volume\n", info.name, info.coarse[0],
         info.coarse[1], info.coarse[2], info.size[0], info.size[1], info.size[2], info.n_start_points, (double)info.stored_bytes / 1024.0,
         (double)nvfx_volume_scratch_bytes(v) / 1048576.0);
  {
    const float controls[3] = {0.7f, 0.5f, 0.6f};
    nvfx_volume_set_controls(v, controls, 3);
  }
  nvfx_volume_set_seed(v, 12345);
  texels = (size_t)info.size[0] * (size_t)info.size[1] * (size_t)info.size[2];
  tex = (uint16_t*)malloc(texels * 2);
  rgba = (uint8_t*)malloc((size_t)w * h * 4);
  if (!tex || !rgba) return 1;
  frames = (int)(seconds * info.fps);
  for (f = 0; f <= frames; ++f) {
    const float a = 6.2831853f * (float)f / (float)(frames > 0 ? frames : 1);
    nvfx_volume_look_at la = {{2.f * sinf(a), 0.9f, 2.f * cosf(a)}, {0.f, 0.6f, 0.f}, {0.f, 1.f, 0.f}, 42.f, (float)w / (float)h, 0.f, 0.f};
    nvfx_volume_camera cam;
    double t0 = seconds_now();
    if (nvfx_volume_step_to(v, (double)f / info.fps) != NVFX_OK) return 1;
    step_time += seconds_now() - t0;
    t0 = seconds_now();
    nvfx_volume_read(v, NVFX_VOLUME_EXTINCTION, NVFX_VOLUME_HALF, tex, texels * 2); /* upload to the GPU here */
    read_time += seconds_now() - t0;
    t0 = seconds_now();
    nvfx_volume_camera_look_at(&la, &cam);
    if (nvfx_volume_render(v, &cam, w, h, rgba, (size_t)w * 4) != NVFX_OK) return 1;
    draw_time += seconds_now() - t0;
  }
  printf("%d frames on %d threads: step %.1f ms, read %.1f ms, %d x %d picture %.1f ms per frame (wall clock; the first includes the warm-up)\n",
         frames + 1, threads, 1000.0 * step_time / (frames + 1), 1000.0 * read_time / (frames + 1), w, h, 1000.0 * draw_time / (frames + 1));
  out = fopen(argv[2], "wb");
  if (!out) return 1;
  fprintf(out, "P6\n%d %d\n255\n", w, h);
  for (f = 0; f < w * h; ++f) { /* premultiplied over a dark background */
    const float alpha = rgba[4 * f + 3] / 255.f;
    int c;
    for (c = 0; c < 3; ++c) {
      const float px = rgba[4 * f + c] + 18.f * (1.f - alpha);
      fputc(px > 255.f ? 255 : (int)px, out);
    }
  }
  fclose(out);
  free(tex);
  free(rgba);
  nvfx_volume_free(v);
  nvfx_volume_effect_free(e);
  return 0;
}
