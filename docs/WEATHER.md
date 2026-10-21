# The weather director in 2D

Status: **done** (21 October 2026). Audience: dev, artists, research. Code: `src/compose/weather.hpp` and
`weather.cpp` (namespace `nfx::compose::wx`), the `weather` statement of the scene language
([COMPOSE.md](COMPOSE.md) §4), the physical keys of `emit rain` and `emit snow`, hail's `speed`, the day's sky
(`frame daylight, overcast, visibility`), the library entry `weather_director` (`library/weather.nvfxl`). Tests:
`tests/test_weather_director.cpp` (28), `nvfx_alloc_test`, and the frozen keyframes of `examples/scenes/weather_day.nvfxs`.
Seeds: 28,500,000 to 28,599,999. No model was trained for it, and no new C call was added.

This is the TODO's "weather director in 2D", ahead of the 3D weather system ([TODO.md](../TODO.md), "Weather
systems"): a small weather state that moves between regimes in the order real weather keeps and drives the layers that
already exist. Every number it hands a layer comes from a relation in the literature (§3) or is a choice this page names
as one (§3.10).

![A day's weather: keyframes at 2, 8, 14, 19.5, 23.1, 27.8, 32.5 and 35.5 s](figures/weather_day_keyframes.png)

*`examples/scenes/weather_day.nvfxs`, 40 s: clear; clouding over; rain; the gust front and a thunderstorm (two of its six
flashes, hail in the heaviest cells, the hills lost in the rain); clearing in reverse, with a rainbow. The video is
`videos/scene_weather_day.mp4`.*

## 1. In brief

- **The state** (`wx::Member`, read by a script as `sky.rain`, ...): cloud cover, rain rate (mm/h), snow rate (mm/h of
  melted water), hail, the wind now and its mean (m/s at 10 m), gustiness, visibility (m), temperature (C), the rate of
  lightning (flashes a minute), the load of dust, and the fog's extinction; and what follows from them: everything's
  extinction, the wind along x, its Beaufort force, a learned module's lean, leaves in motion, dust raised, the daylight,
  the lightning's light now, the number of flashes, where the last struck, the time since, the regime it heads for and
  how far it has come.
- **Seven regimes**: clear, overcast, rain, storm, snow, fog, dust storm. Each is a target state (§2.1), varied by the seed
  each time it is asked for, with weather noise around it.
- **Transitions in the order real weather keeps** (§2.2): clouds build before rain; the gust front (the wind, its gusts,
  the drop in temperature) arrives before the downpour; lightning and hail come with the heaviest rain; it clears in
  reverse. Deterministic from the seed and the targets asked for, continuous at every frame, interruptible.
- **Weather noise** (§2.3): gusts and rain cells are seeded Ornstein-Uhlenbeck processes stepped exactly, the gusts at the
  Kaimal integral time scale of the surface wind, the rain cells log-normal over minutes; lightning is a Poisson process at
  the rate of the moment, a flash 1 to 5 return strokes.
- **Relations from the literature** (§3): Koschmieder's visibility; Marshall and Palmer's drop sizes; Atlas, Srivastava and
  Sekhon's fall speeds (checked against Gunn and Kinzer's measurements); the rain's extinction from the two; Locatelli and
  Hobbs's snowflakes; the Beaufort scale and what the wind does at each force; Kasten and Czeplak's daylight under cloud;
  the CIE overcast sky; IEC 61400-1's Kaimal length scale; a hailstone's drag.
- **It drives the layers that exist** (§4): physical rain and snow (sizes, speeds, counts and streaks from the rate),
  hail, the cloud and fog models (opacity, look, lean), the storm layers (a bolt where the flash struck, its lamp, the
  flash), the wind every module feels (its learned `wind` control and a wind field), leaves, and the sky's light.
- **Control** (§5): a `weather` statement with a schedule in a script; in an engine, an `input` the script hands to the
  director's `regime` (`nvfx_scene_set_input`), the state read back with `nvfx_scene_get_input(scene, "sky.rain")`, the
  flashes counted by a rule (`nvfx_scene_rule_state`); on a library stage, the entry's parameter (`nvfx_stage_set_param`).
  No new C call.
- **Deterministic, no allocation per frame, about 1 microsecond a step** (§6).

## 2. The state, the regimes and the transitions

### 2.1 The regimes' targets

Each target is a choice inside what is observed for such weather (the American Meteorological Society's rain classes:
light under 2.5 mm/h, moderate to 7.6, heavy above; the WMO's: fog under 1 km, a dust storm under 1 km and a severe one
under 200 m; the US weather service's snow by visibility, FMH-1), not a fit. The visibility and the daylight follow from
the rest (§3).

| regime | cloud | precipitation (mm/h) | hail | mean wind (m/s; force) | gustiness | temperature (C) | lightning (/min) | fog | dust | rain cells | visibility | daylight |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clear | 0.10 | 0 | 0 | 3 (2.3) | 0.20 | 22 | 0 | 0 | 0 | | 40 km | 1.00 |
| overcast | 0.95 | 0 | 0 | 5 (3.3) | 0.18 | 17 | 0 | 0 | 0 | | 40 km | 0.37 |
| rain | 1 | 6 (moderate) | 0 | 7 (4.1) | 0.22 | 13 | 0 | 0 | 0 | 300 s, 0.35 | 3.2 km | 0.25 |
| storm | 1 | 40 (heavy) | 0.6 | 11 (5.6) | 0.40 | 16 | 20 | 0 | 0 | 90 s, 0.5 | 1.0 km | 0.25 |
| snow | 1 | 1.5 (as snow) | 0 | 4 (2.8) | 0.20 | -4 | 0 | 0 | 0 | 400 s, 0.3 | 550 m | 0.25 |
| fog | 1 | 0 | 0 | 1 (1.1) | 0.10 | 8 | 0 | 26 km^-1 | 0 | | 150 m | 0.25 |
| dust storm | 0.30 | 0 | 0 | 15 (6.9) | 0.30 | 34 | 0 | 0 | 0.6 | | 330 m | 0.99 |

Clear air's own extinction is 0.098 km^-1 (a 40 km day). The storm's 16 C against the clear day's 22 C is the gust
front's cold outflow. 20 flashes a minute is a vigorous storm's total (in-cloud and to ground).

**Seeded variability.** Each request (the start included) varies its regime's target with the seed and the request's
number: cloud +-0.04 (a standard deviation), precipitation x exp(0.25 g), hail x exp(0.2 g), wind x exp(0.15 g),
gustiness x exp(0.1 g), temperature +-1.5 C, lightning x exp(0.3 g), fog x exp(0.25 g), dust x exp(0.2 g), g a
standard normal from a hash. So no two storms of a seed are the same, and a seed's are always the same.

### 2.2 Transitions

A request (`Director::request(to, seconds)`) starts from the state as it is (mid-transition too: nothing jumps) and runs
a progress p from 0 to 1 over the time asked for (at least 1 s). Each quantity moves in a window of p of its own, by a
smoothstep, its window depending on whether it rises or falls:

| quantity | rising | falling |
|---|---|---|
| cloud | 0 to 0.45 | 0.55 to 1 |
| mean wind, gustiness | 0.30 to 0.50 | 0.35 to 0.80 |
| temperature (the air mass, with the front) | 0.30 to 0.55 | 0.30 to 0.55 |
| precipitation (and the rain cells' time and spread) | 0.45 to 0.80 | 0.15 to 0.55 |
| lightning | 0.65 to 0.95 | 0 to 0.30 |
| hail | 0.70 to 0.90 | 0 to 0.25 |
| fog | 0.20 to 0.90 | 0 to 0.60 |
| dust | 0.40 to 0.80 | 0.15 to 0.60 |

Building, the clouds come first, then the gust front, then the rain, then the lightning and the hail; clearing, the
reverse: the lightning stops, the rain eases, the wind drops, the clouds break. What else follows from it, measured by
`WeatherDirector.TransitionsKeepTheOrderOfRealWeather` (each at half its change):
- clear to storm: the wind is at 90 % of its rise while the rain is under a fifth of its own;
- clear to dust storm (a haboob): the wind before the dust it raises;
- clear to snow (a cold front): the clouds, then the cold, then the snow; on the way the precipitation falls as rain,
  then sleet, then snow as the air passes 2 C and 0 C (§2.3);
- fog to clear: the fog lifts into low cloud before the sky clears.

Rain to snow is the fastest change of the state: the temperature crossing 2 to 0 C turns the precipitation to snow,
which hides 13 times what rain does per mm/h (§3.4), so the visibility falls within a second or two of a 20 s
transition. It is continuous (its per-step change halves with the step) but it is the reason the jump test allows the
extinction 4 km^-1 a frame.

### 2.3 Weather noise

Every noise is an Ornstein-Uhlenbeck process (Uhlenbeck and Ornstein 1930) of unit variance stepped exactly
(Gillespie 1996: x' = a x + sqrt(1 - a^2) n, a = exp(-dt / tau)), then smoothed by a first-order filter of s seconds and
brought back to unit variance (an OU smoothed over s has the variance tau / (tau + s)), so the state is smooth from frame
to frame: its per-step change halves when the step does (`WeatherDirector.NothingJumps`). Its random numbers are a hash
of the seed, the step and a channel.

| noise | time scale | smoothing | what it does | measured (`WeatherNoise.*`) |
|---|---|---|---|---|
| gusts | Kaimal's integral time scale L_u / U, L_u = 8.1 x 0.7 z = 56.7 m at z = 10 m (IEC 61400-1), 1 to 30 s | 1 s (gusts are measured as 3-second means) | the wind now = mean x (1 + gustiness x noise) | a storm: the spread over the mean 0.356 for a gustiness of 0.355; autocorrelation at tau (5.4 s) 0.453, the process's 0.451; 0.02 at 8 tau |
| rain cells | the regime's (90 s in a storm, 300 to 400 s in steady rain or snow) | 3 s | precipitation x exp(spread x noise - spread^2 / 2): log-normal, its mean kept | spread of ln R 0.475 for 0.5; autocorrelation at 90 s 0.387 (e^-1 = 0.368, the smoothing adds a little) |
| cloud | 90 s | 5 s | cover +- 0.6 c (1 - c) x noise | |
| temperature | 120 s | 5 s | +- 0.3 C x noise | |
| fog | 40 s | 3 s | extinction x exp(0.2 x noise) | |

**Lightning** is a Poisson process at the rate of the moment, which is the base rate times r^2 / exp(spread^2) (r the
rain cells' factor: the flashes fall in the heavier rain, the mean rate kept). A flash has one return stroke 17 % of the
time and otherwise 2 to 5 (3 on average), 40 to 100 ms apart, the first at full light, the later ones at 0.5 to 0.9 of it,
each decaying over 45 ms (after Rakov and Uman 2003: typically 3 to 5 strokes about 60 ms apart). A new flash takes over
from the last. Measured over 3000 s of storm: 1432 flashes for 1444.5 expected; the rain at the flashes 1.66 times the
storm's mean.

**Hail** falls in the heavy cells only: the regime's hail times smooth((r - 0.9) / 0.6), so none in cells lighter than
0.9 of the storm's mean rate and all of it in those 1.5 times as heavy.

**Rain or snow**: the precipitation's share that falls as snow is a ramp from all rain at 2 C to all snow at 0 C (a simple
partition, not a fit: §3.10).

**Dust** is raised as the cube of the wind (the gusts' factor cubed, normalised: sand's flux goes as u*^3, Bagnold 1941)
and follows it over 5 s (a choice: it mixes and settles).

## 3. The relations

### 3.1 Visibility: Koschmieder

A black object at distance x against the horizon sky shows the contrast C = exp(-sigma x), sigma the extinction; at the
eye's threshold of 2 %, the visibility is V = ln(50) / sigma = 3.912 / sigma (Koschmieder 1924). The director adds the
extinctions of clear air, fog, rain (§3.3), snow (§3.4) and dust and gives V = 3912 / sigma m (sigma in km^-1). The day's
sky (§4.6) uses the same law to veil the hills and the ground, and `WeatherDay.TheHillsFadeIntoTheHazeAsKoschmiederSays`
measures it in the picture: the hills' contrast at 4 km and 2 km against 40 km is exp(-3.912 x 2 km (1/V - 1/40 km)) to
within 0.03.

### 3.2 Drop sizes: Marshall and Palmer

N(D) = N0 exp(-Lambda D), N0 = 8000 m^-3 mm^-1, Lambda = 4.1 R^-0.21 mm^-1, R in mm/h (Marshall and Palmer 1948). The drops
drawn are those from 0.5 to 6 mm (smaller ones are drizzle and invisible at the scales drawn; larger ones break up):
there are (N0 / Lambda)(exp(-0.5 Lambda) - exp(-6 Lambda)) of them a cubic metre, from 251 at 1 mm/h to 1801 at 50 mm/h,
their mean diameter 0.74 to 1.05 mm. Each drop's diameter is drawn by the distribution's inverse from a hash.

**The rain's extinction** follows: drops much larger than the wavelength remove twice their cross-section (the extinction
paradox, van de Hulst 1957), so sigma = 2 integral of (pi / 4) D^2 N(D) dD = pi N0 / Lambda^3 = 0.365 R^0.63 km^-1:

| rain (mm/h) | 1 | 5 | 10 | 25 | 50 | 100 |
|---|---:|---:|---:|---:|---:|---:|
| visibility from the rain alone | 10.7 km | 3.9 km | 2.5 km | 1.4 km | 0.91 km | 0.59 km |

### 3.3 Fall speeds: Atlas, Srivastava and Sekhon

v(D) = 9.65 - 10.3 exp(-0.6 D) m/s, D in mm (Atlas, Srivastava and Sekhon 1973; 0 below 0.03 mm, where the fit turns
negative). Against Gunn and Kinzer's (1949) measured terminal velocities it is within 1.4 %:

| D (mm) | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Atlas (m/s) | 4.00 | 6.55 | 7.95 | 8.72 | 9.14 |
| Gunn and Kinzer (m/s) | 4.03 | 6.49 | 8.06 | 8.83 | 9.09 |

**The two together.** The rain rate implied by Marshall-Palmer drops falling at Atlas's speeds, 6 pi 10^-4 integral of
D^3 v(D) N(D) dD, should give R back. It gives a little more: 18 % at 1 mm/h, 16 % at 10, 13 % at 25, 9 % at 50, 5 % at
100 (`WeatherRelations.MarshallPalmerAndAtlasTogetherGiveBackTheRate`). That is what each was fitted to (Marshall and
Palmer's counts at the ground, Atlas's speeds in still air), and is left as it is: the director draws the distribution
the rate names, not one rescaled to close the loop.

**What the rain drawn does with them** (`emit rain, rate R`, §4.1): each drop falls at its own v(D) times the scale (world
pixels a metre), its streak is the distance it falls while the shutter is open, and the count in the region is the
distribution's number in the slab drawn. Measured in a scene (`WeatherScript.ThePhysicalRainHoldsItsDistribution`, at 2,
20 and 60 mm/h): the drops in the region within 8 % of Marshall-Palmer's number, their mean fall speed within 3 % of
Atlas's mean over the distribution, every streak within 1 % of its fall in half a frame. So heavier rain has more, bigger,
faster drops, and longer streaks.

### 3.4 Snowflakes: Locatelli and Hobbs

For aggregates of unrimed radiating assemblages of dendrites, from 2 to 10 mm (Locatelli and Hobbs 1974, Table 1):
V = 0.8 D^0.16 m/s and M = 0.073 D^1.4 mg, D the maximum dimension in mm. Unrimed aggregates fall at about 1 m/s whatever
their size (0.89 m/s at 2 mm, 1.16 at 10). The flakes' sizes are an assumption: exponential, slope 0.4 mm^-1, on 2 to
10 mm (mean 4.2 mm). Their number follows from the mass flux: S / 3600 kg m^-2 s^-1 (S in mm/h of melted water) over
their mean M V: 475 flakes a cubic metre at 1 mm/h.

**Snow's extinction** is twice the flakes' cross-section with a flake covering 0.3 of the circle around it (an assumption:
aggregates are open; Rasmussen and colleagues 1999 find the visibility at a given rate varying by a factor of 3 to 10
with the snow): 1 mm/h gives 825 m, inside the US weather service's "moderate snow" (0.5 to 1 km, FMH-1); 0.5 mm/h 1.6 km,
3 mm/h 280 m. Per mm/h of water, snow hides about 13 times what rain does.

### 3.5 Hail

A sphere of ice (900 kg m^-3) in air (1.2 kg m^-3) at the drag coefficient of a sphere at high Reynolds number (0.6):
v = sqrt(4 g D rho_ice / (3 C_d rho_air)), 12.8 m/s at 10 mm, 18 m/s at 20 mm. The director gives hail as a share (§2.3);
the scene draws its stones at 12.8 m/s (`emit hail, speed`).

### 3.6 Wind: the Beaufort scale

v = 0.836 B^(3/2) m/s at 10 m (WMO); each whole force falls inside its band of the WMO's table
(`WeatherRelations.TheBeaufortScale`). What the wind does on land at each force, and what the layers show of it:

| force | m/s (formula) | on land (WMO) | what the director's layers do |
|---|---:|---|---|
| 0 calm | 0 | smoke rises vertically | `lean` 0.5: a learned module's `wind` control at still air |
| 1 light air | 0.84 | direction of wind shown by smoke drift, but not by wind vanes | smoke and flames lean a little (`lean` grows with the wind, all the way at a gale) |
| 2 light breeze | 2.36 | wind felt on face; leaves rustle | |
| 3 gentle breeze | 4.34 | leaves and small twigs in constant motion | `leaves` from 0 (force 2) to 1 (force 4): leaves blow |
| 4 moderate breeze | 6.69 | raises dust and loose paper; small branches are moved | `lift` from 0 (force 3.5) to 1 (force 5): dust raised (a scene may lay it with the rain) |
| 5 to 7 | 9.35 to 15.5 | small trees sway; large branches in motion; whole trees in motion | (nothing draws trees) |
| 8 gale | 18.9 | breaks twigs off trees; generally impedes progress | `lean` 1: the full lean |
| 9 to 12 | 22.6 to 34.8 | structural damage, trees uprooted, devastation | |

The leaves and the dust follow the wind now, gusts included: a gust makes the leaves fly.

### 3.7 Daylight under cloud: Kasten and Czeplak

G(N) / G(0) = 1 - 0.75 (N/8)^3.4, N the cloud in oktas (Kasten and Czeplak 1980): `sky.light` = 1 - 0.75 cloud^3.4, 0.93
at half cover, 0.25 overcast. It is an average over cloud types: a thunderstorm's sky is darker than any overcast, and
the scene darkens it further by a choice of its own (§4.6).

### 3.8 The overcast sky

The CIE standard overcast sky: the luminance at elevation theta is (1 + 2 sin theta) / 3 of the zenith's (Moon and Spencer
1942, adopted by the CIE in 1955). The day's sky greys into it with `overcast` (§4.6).

### 3.9 The gusts' time scale

The Kaimal spectrum of IEC 61400-1 (edition 3, Annex B) has the longitudinal integral length scale L_u = 8.1 Lambda_1,
Lambda_1 = 0.7 z below 60 m: 56.7 m at 10 m; for the Kaimal spectrum S(0) = 4 sigma^2 L_u / U, so its integral time scale
is L_u / U (5.7 s at 10 m/s). The OU has a Lorentzian spectrum with the same integral time scale (it falls as f^-2 at high
frequency where Kaimal's falls as f^-5/3, so it has a little less fast gusting).

### 3.10 What is a choice, not a relation

| what | the choice | why |
|---|---|---|
| the regimes' targets (§2.1) | inside the observed classes | they set a regime, not measure one |
| the transitions' windows (§2.2) | their order is the observed one; their widths are not measured | a time-lapse wants them readable; real fronts take tens of minutes |
| the rain cells' times (90 to 400 s) and spreads | of the order of rain gauges' minute-scale variability | not fitted to data |
| cloud, temperature and fog noise | small, slow | decoration around the targets |
| rain to snow between 2 and 0 C | a linear ramp | a common simple partition |
| snowflakes' sizes, their area ratio 0.3 | exponential on 2 to 10 mm, mean 4.2 mm | put 1 mm/h inside FMH-1's moderate snow |
| dust's extinction, 3.912 / 0.2 km^-1 at a load of 1 | the WMO's severe dust storm | the load is in its units, not a mass |
| dust following the wind's cube over 5 s | Bagnold's u*^3, a lag of its own | no measurement of the lag here |
| lightning's stroke light, 45 ms | of the order of a return stroke's luminosity and its continuing current | not measured |
| the drops drawn, 0.5 to 6 mm, in a slab 0.15 to 0.5 m deep | a slab of the near field | how many of the near drops to draw is the artist's |
| the wind at eye height, 0.72 of the 10 m wind (the scene's) | a log profile over grass, z0 = 3 cm, at 2 m | the scene's, not the director's |

## 4. The layers it drives

The director only says how much of each there is; every layer is one the scene language had, wired by expressions of
`sky.*`, so an artist can reweigh any of it. `examples/scenes/weather_day.nvfxs` is the reference wiring;
`weather_director` in `library/weather.nvfxl` the same for a library stage.

### 4.1 Rain and snow by the physics

`emit rain` and `emit snow` take four more keys; with `rate`, the emitter is the director's physics (without it, it is the
emitter it was, to the bit):

| key | meaning | default |
|---|---|---|
| `rate` | mm/h (snow: of melted water): rain's drops sized by Marshall-Palmer (§3.2) and falling at Atlas's speeds (§3.3); snow's flakes by Locatelli and Hobbs (§3.4) | (off) |
| `scale` | world pixels a metre where they fall (the near field: 600 is about 3 m from a 50 mm lens on 1280 px) | 600 |
| `depth` | metres of the slab drawn: the count is the distribution's number in it (unless `density` says) | 0.5 |
| `shutter` | rain: seconds the shutter is open: each streak is the distance its drop falls in it | half a frame (180 degrees) |

`radius` multiplies a drop's or a flake's own width there (D times the scale). The region is filled from the distribution
at once; what feeds it from the top is weighted by fall speed (the faster drops cross it sooner), so the drops in it keep
the distribution. The words reserve nothing: a module may still be named `depth` or `shutter`.

### 4.2 Hail

`emit hail, density 1.6 * sky.hail, speed 12.8 * 150`: the director's share, and the stones' speed (§3.5) at a scale.
`speed` is new on `emit hail` (default 560 world pixels a second, as before).

### 4.3 Clouds and fog

The cloud model's masses show by the cover (each a share of the sky: `opacity smooth((sky.cloud - share) / 0.25)`), grey
as the daylight under them falls (`look fleece to storm by ...`), and lean with the wind (`wind sky.lean`). The fog model's
releases are as thick as the fog's extinction makes a path (`opacity 1 - exp(-0.05 * sky.fog)`: 50 m), or in the day scene
as the whole extinction makes 80 m of air (spray and mist in the downpour).

### 4.4 The storm

The thunderstorm entry's layers, driven: `emit arc` where the last flash struck (`bx = X + 520 * sky.bolt`), the same
channel for each of its strokes (`seed 28500100 + 2 * sky.strike`, `rate 0.001`: a new shape for each flash, not each
frame), brightness `3.4 * sky.flash`; a lamp in the cloud where it struck; `light flash (...) * sky.flash`. A rule
`when sky.since == 0 as thunder, repeat:` fires on each flash (for a game's thunder).

### 4.5 The wind every module feels

A learned module's `wind` control takes `sky.lean` (0.5 still air, its full lean at a gale); a `wind` field pushes the
material a little (`velocity (0.02 * eye, 0), on plume`; strong pushes put a learned fire out, [COMPOSE.md](COMPOSE.md)
§5). Particles drift with it (`wind (eye * scale, 0)`), so rain slants as its drops take the wind's speed.

### 4.6 The sky's light

Three new keys of `frame` (all off by default: the night sky of every scene before, to the bit):

| key | meaning |
|---|---|
| `daylight` | the sun's and the sky's light, 1 a clear day: the sky's blue (deeper overhead), the hills lit and the ground's grass |
| `overcast` | 0 the clear blue sky to 1 the CIE overcast sky (§3.8); the stars hide as it greys |
| `visibility` | metres: Koschmieder's haze veils the hills (2 km away), the ground (500 m at the horizon to 24 m at the bottom of the picture) and the sky (2 km at the horizon to 600 m overhead) |

The day scene writes `daylight sky.light * (1 - 0.55 * dark)` (Kasten and Czeplak, darker still in the storm: a choice),
`overcast sky.cloud`, `visibility sky.visibility`, and lets the camera's exposure follow the daylight halfway in stops
(`exposure 1.05 / sqrt(sky.light)`), as a camera or an eye adapts.

## 5. Control

### 5.1 In a script

```text
weather NAME [, seed S] [, regime R] [, over D] [, direction 1 | -1] [:
  REGIME at T [, over D]
  ...]
```

- `seed`: a whole number to 33,554,432 (seeds pass through a float: above 16,777,216 only even ones; an odd one there is an
  error). `direction`: the wind blows along +x (1) or -x (-1).
- `regime` is where it heads: an expression, re-evaluated each frame, whose regime names are values (clear 0, overcast 1,
  rain 2, storm 3, snow 4, fog 5, dust 6); a change starts a transition over `over` seconds (evaluated then; default 60).
  A scene starts in the regime it gives at t = 0.
- With a schedule (the lines under it, or after the colon separated by `;`), `regime` is where it starts (a constant)
  and each line says from when it heads where; a line's `over` is its own, else the statement's.
- `NAME.member` is a value anywhere (`sky.rain`, `sky.visibility`, ...; §1). The director steps first in each frame,
  before the rules (step 0 of [COMPOSE.md](COMPOSE.md) §4.7), so everything in the frame reads its state for that frame.
- The checker gives the line and the column: a misspelt regime (`'strom' is not a regime ... (did you mean 'storm'?)`),
  a schedule out of order, a scheduled weather's regime that is not a constant, a member it has not (`a weather has no
  'rainn' (did you mean 'rain'?)`), a weather used as a value.

### 5.2 In an engine

No new C call: the script hands an input to the director, and the game sets it.

```text
input weather = 0          # the game's: 0 clear ... 6 dust
input change = 30          # seconds a change takes
weather sky, seed 28500010, regime weather, over change
when sky.since == 0 as thunder, repeat:
```

```c
nvfx_scene_set_input(scene, "weather", 3);            /* a storm, over the next 30 s */
float rain, vis;
nvfx_scene_get_input(scene, "sky.rain", &rain);       /* the weather's members read by NAME.member: mm/h */
nvfx_scene_get_input(scene, "sky.visibility", &vis);  /* m: for the game's fog, AI, audio */
int flashes;
nvfx_scene_rule_state(scene, "thunder", &flashes, NULL);   /* a new flash: play the thunder (light 3 s a km later) */
```

`nvfx_scene_get_input` reads a weather's members by `NAME.member` when no input has that name (the one change of the C
API's behaviour: a name it rejected before). On a library stage, `weather_director` takes `regime` and `change` as
parameters (`nvfx_stage_set_param`), and its members are read as `sN_sky.member` (an instance's names start `sN_`, N its
place among the stage's instances in the order of the rooms: `s0_` for a first room of one); each room has
its own director, which runs from the stage's start whether its instance plays or not (an entry's weather has no
schedule: its times would be the stage's). An entry cannot read another instance, so `weather_director` carries its own
layers rather than driving the other weather entries.

## 6. Determinism, threads, allocation, cost

- **Deterministic**: the same seed and the same requests at the same steps give the same state to the bit
  (`WeatherDirector.ItIsDeterministic`); its random numbers are hashes, never the scene's generator, so a scene draws the
  same random numbers with the director and the layers it drives as without (`WeatherScript.ItKeepsTheScenesRandomNumbers`;
  a director that drives nothing changes no pixel); the frames are the same on 1 and 3 threads and overlapped.
- **The seed of the day scene** (28,500,002) was the second tried: its storm flashes six times; the first twenty seeds
  give 0 to 16 flashes in that storm, 4 to 5 most often. Everything else in the scene is as the first seed had it.
- **No allocation per frame**: the director is a fixed set of floats; `nvfx_alloc_test` plays a scene through every
  regime with every layer the director drives (overlapped and not, with the engine layers and without) and a scene
  driven by an input through the C API, reading the weather back every frame: 0 allocations.
- **Cost**: a step is about 1 microsecond (one core of the shared machine at a load of 17, the best of seven runs of
  300,000 steps). The layers cost what they cost without it: the day scene's median frame is 76 ms on 2 threads of the
  same busy machine (provisional), nearly all of it the cloud, fog, fire and smoke modules and the picture at
  1280 x 720.

## 7. The day scene

`examples/scenes/weather_day.nvfxs`, 40 s at 1280 x 720 and 30 fps, about 100 lines; the effects are the library's
(cloud, fog, fire and smoke). The schedule is a time-lapse (a real day takes hours): overcast at 3 s over 7, rain at 11
over 5, storm at 17 over 4, clear at 30 over 8. The gusts and the rain cells keep their own time scales.

| time | the weather | what shows |
|---|---|---|
| 0 to 3 s | clear: a few fair-weather clouds, a light breeze (force 2 to 3) | blue sky, the campfire's smoke leaning a little |
| 3 to 10 s | clouding over | the cloud masses show one by one as the cover passes their shares, greying as the daylight under them falls; the sky greys into the overcast sky; the exposure opens |
| 11 to 16 s | rain (6 mm/h) | thin, short streaks; the hills grey in the haze (3 km) |
| 17 to 21 s | the storm comes in | the gust front first (the wind to force 5 to 6, leaves flying, the smoke and the flames bent over), then the downpour (40 to 60 mm/h: long slanted streaks, mist low over the ground, the hills gone at 1 km), then lightning |
| 21 to 30 s | the storm | six flashes in all, each a forked bolt where it struck, the cloud lit from inside, hail bouncing in the heaviest cells |
| 30 to 38 s | clearing in reverse | the lightning stops first, the rain eases and stops by 34 s, the wind drops, the clouds break and whiten, a double rainbow from 33.6 s |
| 38 to 40 s | clear | |

**By eye.** The order of things reads: the clouds before the rain, the leaves and the bent smoke before the downpour, the
bolts in its heaviest part, and it all undoing in reverse. The day sky, the overcast greying and the haze over the hills
work as relations should (the hills fade at the visibility the rain leaves). What does not, yet:
- the rain at 600 world pixels a metre with the wind's speed is long, bright and slanted; it is what drops near a camera
  do in a strong wind, but it reads heavier than the rate;
- the cloud model's masses are puffs in a band, not a deck: "overcast" is mostly the sky behind them;
- the storm's sky is a flat dark grey; nothing draws the cloud base, rain shafts in the distance or the anvil;
- the ground does not get wet, puddle or darken in the rain, and nothing draws trees for the Beaufort forces 5 to 9;
- the flash lights the scene evenly (the frame's flash): there is no shadow or direction to it;
- the rainbow is the procedural one, placed by the script, not by the sun's direction; nothing in the scene says where the
  sun is.

## 8. Tests

`tests/test_weather_director.cpp`, its own program (it counts allocations):
- **the relations** (`WeatherRelations.*`): Koschmieder (and every frame's visibility against its extinction);
  Marshall-Palmer (the number, the mean and the sampling against numerical integrals, the extinction against geometric
  optics); Atlas against Gunn and Kinzer; the two together giving back the rate; the Beaufort bands and effects; Kasten and
  Czeplak; Locatelli and Hobbs (the number against the mass flux, FMH-1's moderate snow); hail's drag;
- **the noise** (`WeatherNoise.*`): the gusts' mean, spread and autocorrelation against the OU's; the rain cells' spread and
  e-folding time; lightning's Poisson count and the flashes in the heavier rain; hail only in heavy cells, snow only in the
  cold;
- **the director** (`WeatherDirector.*`): determinism; no jumps (the most each member moves in a frame through every
  transition and two interrupted ones, and halving at 60 fps); each regime reaching its targets (exactly, and on average
  over an hour of noise, for four seeds); the order of transitions; a request starting from where it is;
- **the script** (`WeatherScript.*`): the parser, the printer and the checker's messages; the scene and the engine reading
  the weather (an input drives it, `nvfx_scene_get_input` reads it, a rule counts the flashes); the physical rain's
  distribution, streaks and counts; snow and hail; the scene's random numbers kept; the same frames on any number of
  threads;
- **the library** (`WeatherLibrary.*`): an entry's weather renamed with its room, two rooms with their own weather, a
  parameter set through the C API, a schedule refused;
- **the day** (`WeatherDay.*`): the night as it was, to the bit; the hills' contrast against Koschmieder;
- **no allocation** (`WeatherAllocation.NoneAFrame`).

And in ctest: `neuralfx.ScriptedWeatherDayFrozen` (the scene's eight keyframes against their SHA-256 in
`results/experiments/weather_frozen.csv`, float on the AVX2 build as the other frozen tests; skipped, exit 77, without the
library's models) and `neuralfx.SceneScriptChecks.weather_day`.

## 9. Limits

- **It is a director, not a weather model.** Nothing is simulated: no cloud physics, no fronts, no terrain, no diurnal
  cycle, no humidity or dew point. The 3D weather system of the TODO (regional simulations learned as a coarse 3D rollout)
  is what would make the targets come from physics; this makes their local effects coherent and ordered meanwhile.
- **2D and one place**: the state is the weather at the camera, the same everywhere in the picture; a storm does not move
  across the scene, nor a rain shaft stand in the distance (the learned rain shaft could, wired by hand).
- **The transitions' windows and the regimes' targets are choices** (§3.10): the order is the observed one, the
  durations are whatever the script asks.
- **The relations hold where they were measured**: Marshall-Palmer for widespread rain (it overestimates small drops,
  and convective rain is better fitted by other N0); Atlas's speeds near sea level (drops fall faster in thinner air);
  Locatelli and Hobbs for one kind of aggregate; Kasten and Czeplak averaged over cloud types; Koschmieder for a black
  object against the horizon sky by day. The rain implied by the first two together is 5 to 18 % more than the rate
  (§3.3).
- **Snow's visibility rests on an assumed area ratio** (0.3) and an assumed size distribution, set so that 1 mm/h is
  FMH-1's moderate snow; real snow varies by a factor of 3 to 10 at a given rate.
- **The day sky is not a sky model**: no sun, no direction to the light, no colour of the hour; the overcast sky is the
  CIE's gradient in a fixed grey; dust does not dim the daylight (a haboob's darkness is not drawn).
- **Nothing settles or wets**: rain makes no splashes or puddles, snow does not lie, the ground does not darken; weather
  particles are unlit ([COMPOSE.md](COMPOSE.md) §4.9).
- **A flash lights the scene evenly**; the bolt's place is random across the scene and lightning does not ignite fires.
- **The engine reads the state by name** (`nvfx_scene_get_input`), one value a call; the flashes by a rule's count.
- **Library rooms have weather each**: two rooms of `weather_director` on a stage are two weathers.

## References

- Atlas, D., Srivastava, R. C. and Sekhon, R. S. (1973). Doppler radar characteristics of precipitation at vertical
  incidence. *Reviews of Geophysics and Space Physics* 11, 1-35.
- Bagnold, R. A. (1941). *The Physics of Blown Sand and Desert Dunes*. Methuen.
- Gillespie, D. T. (1996). Exact numerical simulation of the Ornstein-Uhlenbeck process and its integral. *Physical
  Review E* 54, 2084-2091.
- Gunn, R. and Kinzer, G. D. (1949). The terminal velocity of fall for water droplets in stagnant air. *Journal of
  Meteorology* 6, 243-248.
- IEC 61400-1, edition 3 (2005). Wind turbines, Part 1: Design requirements. Annex B, the Kaimal spectrum.
- Kaimal, J. C., Wyngaard, J. C., Izumi, Y. and Coté, O. R. (1972). Spectral characteristics of surface-layer turbulence.
  *Quarterly Journal of the Royal Meteorological Society* 98, 563-589.
- Kasten, F. and Czeplak, G. (1980). Solar and terrestrial radiation dependent on the amount and type of cloud. *Solar
  Energy* 24, 177-189.
- Koschmieder, H. (1924). Theorie der horizontalen Sichtweite. *Beiträge zur Physik der freien Atmosphäre* 12, 33-53 and
  171-181.
- Locatelli, J. D. and Hobbs, P. V. (1974). Fall speeds and masses of solid precipitation particles. *Journal of
  Geophysical Research* 79, 2185-2197.
- Marshall, J. S. and Palmer, W. McK. (1948). The distribution of raindrops with size. *Journal of Meteorology* 5, 165-166.
- Moon, P. and Spencer, D. E. (1942). Illumination from a non-uniform sky. *Illuminating Engineering* 37, 707-726 (the CIE
  standard overcast sky, 1955).
- Office of the Federal Coordinator for Meteorology (FMH-1). *Federal Meteorological Handbook No. 1: Surface Weather
  Observations and Reports*: snow's intensity by visibility.
- Rakov, V. A. and Uman, M. A. (2003). *Lightning: Physics and Effects*. Cambridge University Press.
- Rasmussen, R. M., Vivekanandan, J., Cole, J., Myers, B. and Masters, C. (1999). The estimation of snowfall rate using
  visibility. *Journal of Applied Meteorology* 38, 1542-1563.
- Uhlenbeck, G. E. and Ornstein, L. S. (1930). On the theory of the Brownian motion. *Physical Review* 36, 823-841.
- van de Hulst, H. C. (1957). *Light Scattering by Small Particles*. Wiley (the extinction paradox).
- World Meteorological Organization. *Manual on Codes* (WMO-No. 306) and the Beaufort scale (v = 0.836 B^(3/2) m/s);
  *International Cloud Atlas*: fog and dust storm by visibility.
