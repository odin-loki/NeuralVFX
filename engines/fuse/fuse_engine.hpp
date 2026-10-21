// What FUSE implements for NeuralVFX: the engine-facing interface of the sample integration (docs/FUSE.md §2).
//
// There is no FUSE source in this repository. The adapter (nvfx_fuse.hpp) is written against these few interfaces,
// which FUSE implements over its own systems; engines/fuse/demo implements them with a stand-in (a thread pool, a
// render thread and a software framebuffer), so that the whole integration runs headless.
//
//   FrameClock   the game's clock: frame index, time and dt, read once per game frame
//   JobSystem    FUSE's workers: run N jobs and wait (the manager's jobs, the 2D scenes, the volume)
//   View         the camera of a frame: world to screen, its pose, and its matrices (for the volume's renderer)
//   GpuUpload    the texture upload callback: textures made at load, updated on the render thread
//   RenderHooks  a callback FUSE's render thread makes once per frame it draws, with a DrawSink for the quads
//
// Rules every implementation keeps: nothing here may allocate in a frame (JobSystem::run, View's calls, the render
// hook, GpuUpload::update and DrawSink::draw are called every frame); a job is a plain function and a context pointer,
// never a std::function. World units are FUSE's (metres in the demo), y up.
#pragma once

#include <cstddef>
#include <cstdint>

namespace fuse {

struct Vec3 {
  float x = 0.f, y = 0.f, z = 0.f;
};

// --- the frame clock ---------------------------------------------------------------------------------------------------

struct FrameTime {
  std::uint64_t frame = 0;  // 0, 1, 2, ...: the game frame being updated
  double time = 0.0;        // game seconds (pauses and slow motion included: the effects follow this clock)
  double dt = 0.0;          // seconds since the last frame (>= 0; 0 while paused)
};

class FrameClock {
 public:
  virtual ~FrameClock() = default;
  virtual FrameTime now() const noexcept = 0;  // the frame the game thread is updating
};

// --- the job system ----------------------------------------------------------------------------------------------------

class JobSystem {
 public:
  using Job = void (*)(void* context, int job) noexcept;
  virtual ~JobSystem() = default;
  // Threads that may run jobs at once, the calling thread included if it helps.
  virtual int concurrency() const noexcept = 0;
  // Run fn(context, j) for every j in [0, jobs) on FUSE's workers and return when all have returned. Jobs should be
  // started in index order (the adapter puts its longest first). No allocation. Called from the game thread only.
  virtual void run(int jobs, Job fn, void* context) noexcept = 0;
};

// --- the view: world to screen -----------------------------------------------------------------------------------------

struct ScreenPoint {
  float x = 0.f, y = 0.f;   // pixels from the viewport's top-left corner, y down
  float depth = 0.f;        // view-space distance along the camera's axis (> 0: in front of the camera)
};

struct CameraPose {
  Vec3 eye, forward, up;    // forward and up are unit vectors, at right angles
  float fov_y = 1.f;        // vertical field of view, radians
};

class View {
 public:
  virtual ~View() = default;
  virtual int width() const noexcept = 0;  // the viewport, pixels
  virtual int height() const noexcept = 0;
  virtual ScreenPoint world_to_screen(const Vec3& p) const noexcept = 0;
  virtual CameraPose pose() const noexcept = 0;
  // The same camera as column-major OpenGL-style matrices: world to view, and view to clip (the camera looks down its
  // -z axis, clip z in [-1, 1]). FUSE converts its own conventions here; the volume's renderer takes these.
  virtual void matrices(float world_to_view[16], float view_to_clip[16]) const noexcept = 0;
};

// --- textures: the upload callback -------------------------------------------------------------------------------------

using Texture = std::uint32_t;  // FUSE's handle; 0 is none

enum class TextureFormat : std::uint8_t {
  Rgba8Srgb,   // 4 bytes a texel, premultiplied; rgb decoded as sRGB (alpha linear) before filtering and blending
  R16Float3d,  // 2 bytes a texel (IEEE half), a 3D texture: a volume's field for FUSE's own ray marcher
};

struct TextureDesc {
  int width = 0, height = 0, depth = 1;  // depth > 1 only for 3D formats
  TextureFormat format = TextureFormat::Rgba8Srgb;
  const char* name = "";                 // for FUSE's tools
};

struct TextureRegion {
  int x = 0, y = 0, z = 0;
  int width = 0, height = 0, depth = 1;
};

class GpuUpload {
 public:
  virtual ~GpuUpload() = default;
  // At load, on the loading thread (FUSE marshals to its render thread if it must). 0: failed.
  virtual Texture create_texture(const TextureDesc& desc) = 0;
  virtual void destroy_texture(Texture texture) noexcept = 0;
  // On the render thread, in a render hook: copy a region from CPU memory (rows `row_bytes` apart, slices
  // `row_bytes * region.height` apart). The memory is valid for this call only. No allocation.
  virtual void update(Texture texture, const TextureRegion& region, const void* pixels, std::size_t row_bytes) noexcept = 0;
};

// --- drawing: what the render hook submits -----------------------------------------------------------------------------

// A textured rectangle in the world: `centre`, and `axis_u` and `axis_v` half its width and height (u to the right of
// the picture, v up it). Every quad blends premultiplied: dst = src.rgb + dst * (1 - src.a) (source ONE, destination
// ONE_MINUS_SRC_ALPHA), texels decoded as their texture's format says, without depth writes; FUSE sorts transparent
// quads back to front by `depth` (and keeps the order of equal depths).
struct Quad {
  Texture texture = 0;
  float u0 = 0.f, v0 = 0.f, u1 = 1.f, v1 = 1.f;  // the part of the texture to draw (v0: the picture's top row)
  Vec3 centre, axis_u, axis_v;
  float depth = 0.f;    // view depth of the centre, for sorting (the adapter fills it from View::world_to_screen)
  float opacity = 1.f;  // multiplies the premultiplied texel (fades)
};

class DrawSink {
 public:
  virtual ~DrawSink() = default;
  virtual void draw(const Quad& quad) noexcept = 0;  // no allocation; FUSE copies the quad
};

// --- the render-thread hook --------------------------------------------------------------------------------------------

struct RenderFrame {
  std::uint64_t frame = 0;  // the game frame being drawn (FUSE draws frame N while the game thread updates N + 1)
  DrawSink* sink = nullptr;
};

class RenderHooks {
 public:
  using Hook = void (*)(void* context, const RenderFrame& frame) noexcept;
  virtual ~RenderHooks() = default;
  // FUSE's render thread calls fn(context, frame) once for every game frame it draws, in order, after the game thread
  // has finished updating that frame, before it draws the frame's transparent quads. Registered at load; false when FUSE
  // has no room for another hook.
  virtual bool add(Hook fn, void* context) = 0;
  // After remove returns, fn is not running and is not called again.
  virtual void remove(Hook fn, void* context) noexcept = 0;
};

// The services the adapter needs, as FUSE hands them over at load. The adapter keeps the pointers: they must outlive it.
struct Engine {
  JobSystem* jobs = nullptr;
  RenderHooks* hooks = nullptr;
  GpuUpload* gpu = nullptr;
};

}  // namespace fuse
