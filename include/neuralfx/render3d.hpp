// Study N's volume renderer (docs/STUDY_3D.md §3): an emission-absorption ray marcher for fields of temperature and
// soot, used for the 3D simulator's pictures, the learned 3D effect's pictures and the volume C API (nvfx_volume.h).
//
// The light:
//   - emission by hot gas and soot, a grey body at the local temperature: Planck's law integrated against the CIE 1931
//     2-degree colour matching functions (in Wyman, Sloan and Shirley's multi-lobe fit, JCGT 2013) to XYZ, then to
//     linear sRGB (D65), tabulated over temperature; the emission coefficient equals the absorption coefficient
//     (Kirchhoff), so a thick isothermal slab shows the blackbody colour;
//   - absorption and single scattering by soot (isotropic) of a fixed key light, shadowed by a sweep through the
//     volume towards the light, an ambient term, and light from the fire itself (its emission as a point light at the
//     emission's centre, unshadowed);
//   - one fixed exposure and tone curve: 1 - exp(-exposure L) per channel, then the sRGB transfer curve.
// The output is premultiplied RGBA: rgb is the light the volume adds, alpha one minus its transmittance. Blend with
// dst = src.rgb + dst (1 - src.a).
//
// The volume occupies the world box [-0.5, 0.5] x [0, ny / nx] x [-0.5 nz / nx, 0.5 nz / nx]: one world unit across,
// its floor at y = 0. Fields are nx * ny * nz values indexed (y * nz + z) * nx + x (sim3d.hpp's layout). Cameras are
// view and projection matrices (column-major, OpenGL conventions: the camera looks down its -z axis, clip z in
// [-1, 1]), so any camera can orbit the effect.
//
// Deterministic: each pixel is computed alone, the preparation per cell, sums in a fixed order; the same pictures on
// any number of threads. No allocation after construction.
#pragma once

#include <neuralfx/parallel.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace nfx::render3d {

struct Optics {
  float kelvin0 = 900.f;          // temperature 0 of the fields is this many kelvin (the cold gas, which hardly glows)
  float kelvin_per_unit = 1000.f; // kelvin per unit of the fields' temperature
  float soot_extinction = 30.f;   // per world unit, per unit of soot
  float flame_extinction = 14.f;  // per world unit where the gas is hot: hot soot in the flame, times smoothstep(flame0,
  float flame0 = 0.3f, flame1 = 1.0f;   // flame1, temperature)
  float albedo = 0.5f;            // soot's single-scattering albedo
  float emission = 4.f;           // emission scale: a blackbody at reference_kelvin has luminance `emission`
  float reference_kelvin = 1800.f;
  std::array<float, 3> light_dir{-0.40f, 0.85f, 0.35f};  // towards the key light (normalised when used)
  std::array<float, 3> light_rgb{1.10f, 1.06f, 1.00f};   // the key light's in-scattered radiance (phase included)
  std::array<float, 3> ambient_rgb{0.10f, 0.11f, 0.14f};
  float fire_light = 0.6f;        // light from the fire on its smoke, as a point light at the emission's centre
  float exposure = 1.f;           // then 1 - exp(-exposure L), then the sRGB curve
  float step = 0.6f;              // ray-march step, in cells
};

// The blackbody colour table: linear sRGB radiance of a blackbody at `kelvin`, normalised so that the luminance at the
// optics' reference temperature is 1 (negative components, outside sRGB's gamut, are clamped to 0).
std::array<float, 3> blackbody_rgb(float kelvin);

struct Camera {
  std::array<float, 16> view{};  // column-major
  std::array<float, 16> proj{};
};
// A perspective camera looking from `eye` at `target` (world units), vertical field of view in degrees.
Camera look_at(std::array<float, 3> eye, std::array<float, 3> target, std::array<float, 3> up, float fov_y_degrees, float aspect,
               float near_plane = 0.02f, float far_plane = 50.f);
// An orthographic camera (for tests): looks along -z of its view, showing `half_width` x `half_height` world units
// either side of its axis.
Camera orthographic(std::array<float, 3> eye, std::array<float, 3> target, std::array<float, 3> up, float half_width, float half_height,
                    float near_plane = 0.01f, float far_plane = 50.f);

// Whether a camera's matrices are finite and proj * view can be inverted (render() throws otherwise).
bool invertible(const Camera& cam);

// The renderer for one grid size. prepare() turns the fields into the volume the rays read (per cell: extinction and the
// light each cell adds); render() draws it for a camera. Construction allocates everything.
class Renderer {
 public:
  Renderer(int nx, int ny, int nz, const Optics& optics = {});
  int nx() const { return nx_; }
  int ny() const { return ny_; }
  int nz() const { return nz_; }
  const Optics& optics() const { return o_; }
  std::size_t bytes() const;

  // temperature and soot: nx * ny * nz each. A pool, if given, shares the work; the result is the same without one.
  void prepare(std::span<const float> temperature, std::span<const float> soot, Pool* pool = nullptr);
  // Premultiplied RGBA8 after the exposure and tone curve, rows top to bottom, `stride` bytes apart.
  void render(const Camera& cam, int width, int height, std::uint8_t* rgba, std::size_t stride, Pool* pool = nullptr) const;
  // The same before the tone curve: linear radiance (premultiplied) and alpha, 4 floats per pixel, rows top to bottom.
  void render_linear(const Camera& cam, int width, int height, float* rgba, Pool* pool = nullptr) const;
  // The prepared extinction (per world unit), nx * ny * nz.
  std::span<const float> extinction() const { return sigma_; }

 private:
  template <bool kTone>
  void render_rows(const std::array<double, 16>& inverse_view_proj, int width, int height, int y0, int y1, void* out, std::size_t stride) const;
  int nx_, ny_, nz_;
  Optics o_;
  std::array<float, 3> table_scale_{};
  std::vector<float> table_;  // blackbody rgb per kelvin step
  std::vector<float> sigma_, tlight_, vol_;  // extinction; transmittance towards the light; [cell][4] (sigma, light rgb)
  std::vector<float> macro_;                 // max extinction per macro cell (kMacro^3 cells, with a cell of margin)
  int mx_ = 0, my_ = 0, mz_ = 0;
  std::vector<double> sums_;                 // per slab: emitted power and its centre, for the fire's light
};

inline constexpr int kMacro = 4;

}  // namespace nfx::render3d
