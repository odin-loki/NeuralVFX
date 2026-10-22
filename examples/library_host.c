/* nvfx_library_host: how an engine uses the effect library, in plain C (include/neuralfx/nvfx_library.h,
 * docs/LIBRARY.md).
 *
 *   nvfx_library_host CATALOGUE_DIR MODELS_DIR [--seconds 4] [--out frame.pam]
 *       list the catalogue, make a stage with room for a few effects, spawn them, drive them while the stage plays
 *       (a parameter, a move, a stop), print the cost per frame and the heat under each, and write the last frame
 *   nvfx_library_host --self-test CATALOGUE_DIR
 *       check the API without any learned effect: the catalogue's listing, its errors (file, line and column), and a
 *       stage of procedural effects spawned, driven, stopped, stolen and cleared as a game would, its frames the same
 *       for the same calls (exit status: the number of failed checks)
 */
#include <neuralfx/nvfx_library.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static int failures = 0;
#define CHECK(cond)                                      \
  do {                                                   \
    if (!(cond)) {                                       \
      fprintf(stderr, "self-test failed: %s\n", #cond); \
      ++failures;                                        \
    }                                                    \
  } while (0)

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

static void list(const nvfx_library* lib) {
  int i, c;
  printf("%d effects in %d categories\n", nvfx_library_count(lib), nvfx_library_category_count(lib));
  for (c = 0; c < nvfx_library_category_count(lib); ++c) {
    const char* cat = nvfx_library_category(lib, c);
    printf("  %s:", cat);
    for (i = 0; i < nvfx_library_count(lib); ++i) {
      nvfx_library_effect e;
      nvfx_library_get_effect(lib, i, &e);
      if (strcmp(e.category, cat) == 0) printf(" %s", e.name);
    }
    printf("\n");
  }
}

/* The frames of a stage of procedural effects driven by a fixed list of calls: its checksum (FNV-1a over the RGBA). */
static unsigned long long play_procedural(const nvfx_library* lib, const char* a, const char* b, int frames, int threads) {
  nvfx_stage_room rooms[2];
  nvfx_stage_desc sd;
  nvfx_stage* stage = NULL;
  nvfx_scene_error err;
  nvfx_scene_info info;
  nvfx_fx fa = 0, fb = 0;
  unsigned char* rgba;
  unsigned long long h = 1469598103934665603ull;
  int f;
  size_t i;
  rooms[0].effect = a;
  rooms[0].count = 2;
  rooms[0].steal = 1;
  rooms[1].effect = b;
  rooms[1].count = 1;
  rooms[1].steal = 0;
  nvfx_stage_desc_init(&sd);
  sd.width = 320;
  sd.height = 200;
  sd.rooms = rooms;
  sd.n_rooms = 2;
  sd.threads = threads;
  if (nvfx_stage_create(lib, &sd, &stage, &err) != NVFX_OK) {
    fprintf(stderr, "stage: %s\n", err.message);
    ++failures;
    return 0;
  }
  nvfx_scene_get_info(nvfx_stage_scene(stage), &info);
  rgba = (unsigned char*)malloc((size_t)info.width * (size_t)info.height * 4);
  nvfx_stage_spawn(stage, a, 100.f, 170.f, 1.f, 0.2, &fa);
  nvfx_stage_spawn(stage, b, 220.f, 120.f, 0.8f, 0.5, &fb);
  for (f = 0; f < frames; ++f) {
    if (f > 0) nvfx_scene_step_frames(nvfx_stage_scene(stage), 1);
    if (f == 20) nvfx_stage_move(stage, fa, 120.f, 170.f);
    if (f == 30) nvfx_stage_stop(stage, fb, -1.0);
    nvfx_scene_render(nvfx_stage_scene(stage), rgba, (size_t)info.width * 4);
  }
  for (i = 0; i < (size_t)info.width * (size_t)info.height * 4; ++i) h = (h ^ rgba[i]) * 1099511628211ull;
  free(rgba);
  nvfx_stage_free(stage);
  return h;
}

static int self_test(const char* dir) {
  nvfx_library* lib = open_library(dir, NULL);
  nvfx_library_desc d;
  nvfx_library_file file;
  nvfx_library* bad = NULL;
  nvfx_scene_error err;
  const char* proc[2] = {NULL, NULL}; /* two procedural effects (no learned effect needed) */
  int i, n_proc = 0;
  CHECK(lib != NULL);
  if (!lib) return failures;
  list(lib);
  CHECK(nvfx_library_count(lib) >= 30);
  CHECK(nvfx_library_find(lib, "campfire") >= 0 && nvfx_library_find(lib, "no such effect") == -1);
  for (i = 0; i < nvfx_library_count(lib); ++i) {
    nvfx_library_effect e;
    CHECK(nvfx_library_get_effect(lib, i, &e) == NVFX_OK);
    CHECK(e.name && e.name[0] && e.description[0] && e.width > 0.f && e.height > 0.f);
    CHECK(e.looping || e.duration > 0.f);
    if (e.models[0] == '\0' && n_proc < 2 && (strcmp(e.name, "sparks") == 0 || strcmp(e.name, "lightning_strike") == 0)) proc[n_proc++] = e.name;
    if (e.n_params > 0) {
      nvfx_library_param p;
      CHECK(nvfx_library_get_param(lib, i, 0, &p) == NVFX_OK && p.min <= p.value && p.value <= p.max && p.description[0]);
      CHECK(nvfx_library_get_param(lib, i, e.n_params, &p) == NVFX_ERROR_ARGUMENT);
    }
  }
  CHECK(nvfx_library_get_effect(lib, -1, NULL) == NVFX_ERROR_ARGUMENT);
  CHECK(nvfx_library_category(lib, 0) != NULL && strcmp(nvfx_library_category(lib, 0), "fire") == 0);
  CHECK(nvfx_library_category(lib, 99) == NULL);

  /* catalogue errors: the file, the line and the column */
  nvfx_library_desc_init(&d);
  file.name = "broken.nvfxl";
  file.text = "effect fire = \"fire.nvfx\"\nentry torch, category fire, size 90 x 190, loop, about \"x\"\n"
              "field warm = lamp, at (X, Y), strenght 1\n";
  d.files = &file;
  d.n_files = 1;
  CHECK(nvfx_library_open(&d, &bad, &err) == NVFX_ERROR_SCRIPT && bad == NULL);
  CHECK(err.line == 3 && err.column == 31 && strncmp(err.message, "broken.nvfxl:3:31:", 18) == 0);
  printf("a broken catalogue: %s\n", err.message);
  file.text = "entry torch, category fire, size 90 x 190, loop, about \"x\"\nscene size 64 x 64\n";
  CHECK(nvfx_library_open(&d, &bad, &err) == NVFX_ERROR_SCRIPT && err.line == 2);
  file.text = "entry torch, category fire, size 90 x 190, about \"x\"\n";
  CHECK(nvfx_library_open(&d, &bad, &err) == NVFX_ERROR_SCRIPT && err.line == 1); /* neither loop nor once */
  CHECK(nvfx_library_open(NULL, &bad, &err) == NVFX_ERROR_ARGUMENT);

  /* a stage of procedural effects, as a game drives it */
  CHECK(n_proc == 2);
  if (n_proc == 2) {
    nvfx_stage_room rooms[2];
    nvfx_stage_desc sd;
    nvfx_stage* stage = NULL;
    nvfx_fx a = 0, b = 0, c = 0, x = 0;
    nvfx_stage_fx_info info;
    float v = 0.f;
    unsigned long long h1, h2;
    rooms[0].effect = proc[0];
    rooms[0].count = 1;
    rooms[0].steal = 0;
    rooms[1].effect = proc[1];
    rooms[1].count = 1;
    rooms[1].steal = 1;
    nvfx_stage_desc_init(&sd);
    sd.width = 320;
    sd.height = 200;
    sd.rooms = rooms;
    sd.n_rooms = 2;
    CHECK(nvfx_stage_create(lib, &sd, &stage, &err) == NVFX_OK);
    if (stage) {
      nvfx_scene* scene = nvfx_stage_scene(stage);
      char script[64];
      CHECK(nvfx_stage_script(stage, script, sizeof script) > sizeof script && strlen(script) == sizeof script - 1);
      CHECK(nvfx_stage_free_rooms(stage, proc[0]) == 1 && nvfx_stage_free_rooms(stage, "campfire") == -1);
      CHECK(nvfx_stage_spawn(stage, proc[0], 100.f, 170.f, 1.f, 0.5, &a) == NVFX_OK && a != 0);
      CHECK(nvfx_stage_spawn(stage, proc[0], 100.f, 170.f, 1.f, -1.0, &x) == NVFX_ERROR_MEMORY && x == 0); /* no room */
      CHECK(nvfx_stage_spawn(stage, "campfire", 100.f, 170.f, 1.f, -1.0, &x) == NVFX_ERROR_ARGUMENT);       /* not on it */
      CHECK(nvfx_stage_get_fx(stage, a, &info) == NVFX_OK && info.state == NVFX_FX_WAITING);
      nvfx_scene_step(scene, 0.6, NULL);
      CHECK(nvfx_stage_get_fx(stage, a, &info) == NVFX_OK && info.state == NVFX_FX_PLAYING && fabsf(info.spawned - 0.5f) < 1e-4f);
      CHECK(nvfx_stage_set_param(stage, a, "count", 100.f) == NVFX_OK); /* clamped to its range */
      CHECK(nvfx_stage_get_param(stage, a, "count", &v) == NVFX_OK && v == 30.f);
      CHECK(nvfx_stage_set_param(stage, a, "nonexistent", 1.f) == NVFX_ERROR_ARGUMENT);
      CHECK(nvfx_stage_spawn(stage, proc[1], 200.f, 170.f, 1.f, -1.0, &b) == NVFX_OK);
      nvfx_scene_step(scene, 0.2, NULL);
      CHECK(nvfx_stage_get_fx(stage, b, &info) == NVFX_OK && info.state == NVFX_FX_PLAYING);
      CHECK(nvfx_stage_spawn(stage, proc[1], 220.f, 170.f, 1.f, -1.0, &c) == NVFX_OK && c != b); /* steals b's room */
      CHECK(nvfx_stage_get_fx(stage, b, &info) == NVFX_ERROR_ARGUMENT);                            /* b is stale */
      nvfx_scene_step(scene, 2.0, NULL);
      CHECK(nvfx_stage_get_fx(stage, c, &info) == NVFX_OK && info.state == NVFX_FX_ENDED); /* a one-shot that has played */
      CHECK(nvfx_stage_stop(stage, a, -1.0) == NVFX_OK);
      CHECK(nvfx_stage_get_fx(stage, a, &info) == NVFX_OK && info.state == NVFX_FX_ENDED);
      CHECK(nvfx_stage_free_rooms(stage, proc[0]) == 1);
      CHECK(nvfx_stage_set_camera(stage, 10.f, 0.f) == NVFX_OK);
      CHECK(nvfx_stage_clear(stage) == NVFX_OK && nvfx_scene_time(scene) == 0.0);
      CHECK(nvfx_stage_get_fx(stage, a, &info) == NVFX_ERROR_ARGUMENT);
      nvfx_stage_free(stage);
    }
    /* the same calls give the same frames, on any number of threads */
    h1 = play_procedural(lib, proc[0], proc[1], 40, 1);
    h2 = play_procedural(lib, proc[0], proc[1], 40, 3);
    printf("procedural stage: %016llx on 1 thread, %016llx on 3\n", h1, h2);
    CHECK(h1 == h2 && h1 != 0);
  }
  nvfx_library_free(lib);
  printf("%s: %d failures\n", failures ? "FAILED" : "ok", failures);
  return failures;
}

static int play(int argc, char** argv) {
  nvfx_library* lib = open_library(argv[1], argv[2]);
  const char* out = NULL;
  double seconds = 4.0;
  nvfx_stage_room rooms[4] = {{"campfire", 2, 0}, {"steam_vent", 1, 0}, {"grenade", 4, 1}, {"sparks", 1, 0}};
  nvfx_stage_desc sd;
  nvfx_stage* stage = NULL;
  nvfx_scene* scene;
  nvfx_scene_info info;
  nvfx_scene_error err;
  nvfx_fx fire, vent, spark;
  unsigned char* rgba;
  int i, f, frames;
  double worst = 0.0, total = 0.0;
  for (i = 3; i + 1 < argc; i += 2) {
    if (strcmp(argv[i], "--seconds") == 0) seconds = atof(argv[i + 1]);
    else if (strcmp(argv[i], "--out") == 0) out = argv[i + 1];
  }
  if (!lib) return 1;
  list(lib);
  nvfx_stage_desc_init(&sd);
  sd.rooms = rooms;
  sd.n_rooms = 4;
  if (nvfx_stage_create(lib, &sd, &stage, &err) != NVFX_OK) {
    fprintf(stderr, "stage: %s\n", err.message);
    return 1;
  }
  scene = nvfx_stage_scene(stage);
  nvfx_scene_get_info(scene, &info);
  printf("stage %d x %d, ground at %.0f, %d modules, %.1f MB of working memory\n", info.width, info.height, info.ground, info.n_modules,
         (double)info.scratch_bytes / 1048576.0);
  rgba = (unsigned char*)malloc((size_t)info.width * (size_t)info.height * 4);
  nvfx_stage_spawn(stage, "campfire", 220.f, info.ground, 1.f, -1.0, &fire);
  nvfx_stage_spawn(stage, "steam_vent", 520.f, info.ground, 0.8f, 0.5, &vent);
  nvfx_stage_spawn(stage, "sparks", 760.f, info.ground, 1.f, 1.0, &spark);
  frames = (int)(seconds * info.fps);
  for (f = 0; f < frames; ++f) { /* the game's frame: its own logic, a step and a picture */
    double t0;
    if (f == 30) nvfx_stage_set_param(stage, fire, "intensity", 1.f);
    if (f % 25 == 12) {
      nvfx_fx g;
      nvfx_stage_spawn(stage, "grenade", 380.f + 60.f * (float)(f % 4), info.ground, 0.7f, -1.0, &g); /* four rooms, steals */
    }
    if (f == 60) nvfx_stage_move(stage, vent, 560.f, info.ground);
    if (f == 90) nvfx_stage_stop(stage, spark, -1.0);
    t0 = now_ms();
    if (f > 0) nvfx_scene_step_frames(scene, 1);
    nvfx_scene_render(scene, rgba, (size_t)info.width * 4);
    t0 = now_ms() - t0;
    total += t0;
    if (t0 > worst) worst = t0;
    if (f % 30 == 0) {
      nvfx_scene_fields s;
      nvfx_scene_sample(scene, 220.f, info.ground - 60.f, &s);
      printf("  t %4.1f s: %6.1f ms, heat over the campfire %.2f\n", nvfx_scene_time(scene), t0, s.heat);
    }
  }
  printf("%d frames: %.1f ms a frame on average, %.1f at worst\n", frames, total / frames, worst);
  if (out) {
    FILE* o = fopen(out, "wb");
    if (o) {
      int x, y;
      fprintf(o, "P6\n%d %d\n255\n", info.width, info.height);
      for (y = 0; y < info.height; ++y)
        for (x = 0; x < info.width; ++x) fwrite(rgba + ((size_t)y * (size_t)info.width + (size_t)x) * 4, 1, 3, o);
      fclose(o);
      printf("wrote %s\n", out);
    }
  }
  free(rgba);
  nvfx_stage_free(stage);
  nvfx_library_free(lib);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 3 && strcmp(argv[1], "--self-test") == 0) return self_test(argv[2]);
  if (argc >= 3) return play(argc, argv);
  fprintf(stderr, "usage: %s CATALOGUE_DIR MODELS_DIR [--seconds 4] [--out frame.pam] | --self-test CATALOGUE_DIR\n", argv[0]);
  return 2;
}
