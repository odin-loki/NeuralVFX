# The effect library

A catalogue of 70 named, ready-to-use effects that a game lists, places and drives without training anything: a
campfire, a grenade, a steam vent, a portal, a lightning strike, ground fog, a dust impact, a flamethrower, a plasma
orb, a dust devil. Each plays the eleven trained rollout models (fire, smoke, explosion v2, steam and magic; fog, dust
and cloud, [EFFECTS_ATMOSPHERE.md](EFFECTS_ATMOSPHERE.md); jet, orb and vortex, [EFFECTS_ENERGY.md](EFFECTS_ENERGY.md))
and the compositor's procedural elements (embers, sparks, shock fronts, scorch marks, flashes, lights, bolts, rings), at
a size, with parameters mapped onto the models' controls and looks. The 23 entries of the six newer models were added
with the recipe of §6, in text; the only code they needed is a third anchor, for the jet's nozzle.

![The library's contact sheet: the entries of the first five models](figures/library_sheet.png)

![The library's contact sheet: the 23 entries of fog, dust, cloud, jet, orb and vortex](figures/library_sheet_new.png)

*Every entry alone on a stage fitted to its box, at the moment its catalogue says shows it well (`nvfx_gallery`).*

| category | entries | of them new (round 5's models) | learned | procedural only |
|---|---|---|---|---|
| fire | 14 | 4 | 14 | 0 |
| smoke | 7 | 2 | 7 | 0 |
| explosions | 5 | 0 | 4 | 1 |
| steam (and water) | 5 | 0 | 5 | 0 |
| magic | 8 | 3 | 6 | 2 |
| ambient | 5 | 0 | 3 | 2 |
| sparks (and debris) | 8 | 1 | 2 | 6 |
| light | 7 | 2 | 3 | 4 |
| dust | 5 | 5 | 5 | 0 |
| weather | 6 | 6 | 6 | 0 |
| **all** | **70** | **23** | **55** | **15** |

**What is learned and what is not.** Only eleven things move by a trained model: fire, smoke, an explosion, steam,
magic, fog, dust, cloud, a jet, an orb and a vortex. Everything else in an entry is the compositor's: particles (embers,
sparks, debris), shock fronts, scorch marks, flashes, camera shakes, and the glows this library adds (lightning and
electric arcs, rings, discs and motes of light, lamps). Entries that use no model at all say *(procedural)* in their
description. **A recolour is a look, not new dynamics**: blue gas, green witch fire, the violet magic burst, the red
dark portal, the coloured smokes and fogs, the blue rocket exhaust and the violet dragon breath and magic vortex move
exactly as their model's orange fire, grey smoke, explosion, white fog, jet or dust column does; a field-shader look
draws them in another colour (`hue`, `tint`). Nothing here was trained for the library. Entries marked *(new)* are those
of round 5's six models.

## 1. The catalogue

`library/*.nvfxl`, one file per category plus `models.nvfxl` (the learned effects and shared looks). Sizes are the
effect's box in world pixels at scale 1; *base* anchors stand on the spawn point (bottom centre), *centre* anchors are
centred on it, *left* anchors (the jets) have the spawn point at the middle of the box's left edge, their nozzle.
*Plays* lists the learned models first, then the looks and procedural elements. The last column is how each looks in the
gallery and its video, judged by eye: **good** (usable as it is), **fair** (usable, with the flaw named), **weak** (kept
because it is still the best the models give for it, flaw named). Entries that looked bad were changed (start points,
sizes, smoke placed where the flames are) or replaced: §8 says which.

### Fire (14)

| effect | what it is | plays | box, anchor | runs | parameters (default) | how it looks |
|---|---|---|---|---|---|---|
| `campfire` | A campfire: three flames of the fire model over glowing coals, embers and a warm light on the ground | fire; embers, lamp, scorch | 240 x 300, base | loop | intensity (0.75), wind (0.5) | Good: three flames read as one fire; the embers, the scorch and the warm light on the ground carry it. |
| `torch` | A torch's flame, small and lively, with a few embers; its anchor is the torch's head | fire; embers, lamp | 100 x 190, base | loop | intensity (0.6), wind (0.5) | Good: a small lively flame. |
| `candle` | A candle flame: the fire model at its smallest and calmest, with a soft light | fire; lamp | 40 x 70, base | loop | intensity (0.3) | Fair: at 40 pixels the model's flame is a few pixels wide and flickers more than a candle does. |
| `brazier` | A brazier: a strong, tall flame that throws up a shower of embers now and then; its anchor is the bowl's rim | fire; embers, lamp, burst | 200 x 300, base | loop | intensity (0.85) | Good (after moving its main flame to the calm start point); its ember gusts every 1.9 s are regular. |
| `bonfire` | A bonfire: five flames of the fire model in a heap, a column of smoke above, a storm of embers | fire, smoke; look smoke, embers, lamp, scorch | 400 x 560, base | loop | intensity (0.9), wind (0.5) | Good: the smoke rises out of the flames; heavy (six modules, about 40 ms a frame on one thread). |
| `wall_of_flames` | A wall of flames: nine flames of the fire model in a row, 900 world pixels wide, embers flying | fire; embers, lamp | 900 x 360, base | loop | intensity (0.9), wind (0.5) | Fair: reads as nine tongues side by side rather than one wall of fire. |
| `burning_ground` | Burning ground: a carpet of low flames (spilt oil, a burnt field), eight small flames of the fire model | fire; embers, lamp, scorch | 640 x 140, base | loop | intensity (0.45) | Fair: sparse low flames over glowing patches; better in the distance than close up. |
| `blue_gas_flame` | A blue gas flame: the fire model, calm, recoloured blue by a field-shader look, with a blue light | fire; look gas, lamp | 120 x 190, base | loop | intensity (0.55) | Good: a believable blue flame (the orange fire's motion, recoloured). |
| `witch_fire` | Witch fire: two flames of the fire model recoloured green by a field-shader look, green motes, a green light | fire; look witch, glow, lamp | 220 x 300, base | loop | intensity (0.75) | Good: green flames and motes (a recolour). |
| `burning_wreck` | A burning wreck: a strong, ragged fire under a column of dark smoke, embers and a red light | fire, smoke; look sooty, embers, lamp | 340 x 520, base | loop | intensity (0.8), wind (0.6) | Good: dark smoke rising out of a ragged fire; the smoke is thin at 128 pixels. |
| `flamethrower` (new) | A flamethrower: a roaring jet of burning gas from its nozzle (the anchor), widening, breaking up and curling into smoke; it blows to the right | jet; lamp | 520 x 300, left | loop | intensity (0.8), aim (0) | Good: a roaring jet that widens, breaks up and curls into smoke at its end; it blows to the right only. |
| `gas_leak_fire` (new) | A burning gas leak: a roaring, flickering jet of flame from a broken pipe (the anchor), angled up; it blows to the right | jet; lamp | 300 x 240, left | loop | intensity (0.55) | Good: a small, flickering jet angled up. |
| `rocket_thruster` (new) | A rocket thruster: a fierce blue-white exhaust with a white-hot core at the nozzle (the anchor), the jet recoloured; it blows to the right | jet; look thrust, glow, lamp | 420 x 200, left | loop | intensity (0.9) | Fair: a blue exhaust behind a white-hot core, but its tail curls up as a flame does rather than streaming straight. |
| `fire_whirl` (new) | A fire whirl: a spinning column of flame rising from burning ground, swaying (the vortex model, its warm air drawn as flame) | whirlwind; look whirl_fire, embers, lamp | 320 x 480, base | loop | intensity (0.85), sway (0.7) | Fair: a thin swaying column of flame under a wide orange haze, which can show its tile's square edge; its column is braided (the detail layer's once-a-frame lock, REPORT §12.1). |

### Smoke (7)

| effect | what it is | plays | box, anchor | runs | parameters (default) | how it looks |
|---|---|---|---|---|---|---|
| `chimney_smoke` | Chimney smoke: a thin grey plume that leans with the wind; its anchor is the chimney's top | smoke; look smoke | 260 x 440, base | loop | intensity (0.35), wind (0.65) | Good: a thin leaning plume. |
| `smoke_column` | A column of smoke: the smoke model at its largest, rising from its base (a distant fire, a burning town) | smoke; look smoke | 560 x 680, base | loop | intensity (0.8), wind (0.5) | Fair: a large plume, but wispy rather than a dense column, and its swirly detail shows at this size. |
| `smoke_grenade` | A smoke grenade: a hiss of sparks, then thick white smoke billows out low for eight seconds and thins | smoke; look smoke_white, ceiling, sparks | 520 x 360, base | once, 12 s | intensity (0.9) | Fair: billows and spreads under its ceiling, but late in its life the cloud turns streaked and marbled, and it rises more than grenade smoke. |
| `black_oil_smoke` | Black oil smoke: a dark, rolling column of smoke over burning oil, lit red from below by its flames | smoke, fire; look oil, lamp | 520 x 620, base | loop | intensity (1), wind (0.55) | Fair: dark, and lit red from below, but thinner than real oil smoke: the model's plume is narrow. |
| `poison_gas` (new) | Poison gas: a heavy green gas that pours out low and rolls along the ground (the fog model, recoloured) | fog; look toxic, lamp | 600 x 120, base | loop | intensity (0.8), wind (0.5) | Good: a heavy green gas that stays on the ground (it replaced poison_smoke, whose smoke always rose); its three releases show as humps. |
| `signal_smoke` | Signal smoke: a plume of red smoke from a flare on the ground | smoke; look signal, glow, lamp | 300 x 480, base | loop | intensity (0.6), wind (0.6) | Good: a red plume over a glowing flare. |
| `smoke_screen` (new) | A smoke screen: a thick white wall of smoke that lingers where it was laid and boils slowly (the cloud model) | cloud; look screen | 700 x 240, base | loop | intensity (0.9), wind (0.5) | Good: a thick white wall of smoke that lingers and boils. |

### Explosions (5)

| effect | what it is | plays | box, anchor | runs | parameters (default) | how it looks |
|---|---|---|---|---|---|---|
| `grenade` | A grenade: a ball of fire from the explosion model, a shock front, a flash, embers, debris and a scorch mark | explosion; look cloud, flash, shake, shock, scorch, burst | 340 x 360, base | once, 3 s | power (1) | Good: fireball, sparks, shock front, flash and scorch; the best of the one-shots. |
| `large_blast` | A large blast: a big fireball that rises and cools, handed over to the smoke model, which carries the smoke on | explosion, smoke; look cloud, look smoke, flash, shake, shock, scorch, burst | 640 x 700, base | once, 9 s | power (1) | Fair: a big blobby fireball with swirl patterns on its surface; the hand-over to the smoke model works. |
| `magic_burst` | A magic burst: the explosion model recoloured violet, a ring of light, a violet flash and glowing motes | explosion; look burst_violet, flash, ring, glow, shock | 360 x 360, centre | once, 2.2 s | power (1) | Good: a violet ball (the explosion, recoloured), a ring and motes. |
| `teleport_flash` | A teleport flash: a column of light, a flash, an expanding ring on the ground and sparkles (procedural) | arc, glow, ring, lamp, flash | 240 x 360, base | once, 1.2 s | hue_shift (0) | Good: a clean column of light and a ring. |
| `airburst` | An airburst: an explosion in the air (no scorch), its embers falling as it fades | explosion; look cloud, flash, shake, shock, burst | 380 x 380, centre | once, 3 s | power (1) | Good: a fireball in the air with sparks flying out. |

### Steam and water (5)

| effect | what it is | plays | box, anchor | runs | parameters (default) | how it looks |
|---|---|---|---|---|---|---|
| `steam_vent` | A steam vent: a jet of white vapour from a hole in the ground, hissing unevenly | steam; look vapour | 280 x 400, base | loop | intensity (0.6), wind (0.5) | Fair: a white jet that breaks into puffs; how it looks depends on the seed (the shards' healing replaces a jet that dies). |
| `geyser` | A geyser: a gentle hiss that erupts into a tall column of steam every eight seconds, then dies down | steam; look vapour | 420 x 700, base | loop | period (8) | Fair: the eruption reads; the column is wispy at the top. |
| `pipe_leak` | A leaking pipe: a small, fast jet of steam bent sideways; its anchor is the leak | steam; look vapour | 200 x 200, base | loop | intensity (0.5), wind (0.95) | Good: a bent white jet. |
| `kettle_steam` | Steam from a kettle or a cooking pot: a small, slow wisp; its anchor is the spout | steam; look vapour | 90 x 140, base | loop | intensity (0.3) | Good: a small wisp. |
| `hot_springs` | Hot springs: wisps of steam rising from pools across the ground and drifting with the air, five steam modules | steam; look mist, wind | 700 x 300, base | loop | intensity (0.5) | Fair: five wisps across the ground; it was waterfall mist, renamed because the steam model makes plumes, not a veil. |

### Magic (8)

| effect | what it is | plays | box, anchor | runs | parameters (default) | how it looks |
|---|---|---|---|---|---|---|
| `portal` | A magic portal: the magic model opening over two seconds, spinning, its violet light on everything around | magic; glow, lamp | 440 x 440, centre | loop | spin (0.8), intensity (0.95) | Good: the magic model's spiral at its best; since study L (its magic) smooth arms with a bright rim where they were saw-toothed. |
| `dark_portal` | A dark portal: the magic model recoloured blood red over black mist, crackling with red arcs | magic; look void, arc, lamp | 440 x 440, centre | loop | spin (0.6) | Good: red over black, crackling with arcs (a recolour); since study L a fuller, smoother ring. |
| `healing_circle` | A healing circle: a green ring of light on the ground, motes rising through it and a soft green light (procedural) | ring, glow, lamp | 320 x 260, base | loop | intensity (0.8) | Good: a clean ring and rising motes. |
| `magic_orb` | A magic orb: a small spinning knot of the magic model with a bright core and a violet light | magic; glow, lamp | 140 x 140, centre | loop | spin (0.7) | Good: a small spinning knot with a bright core. |
| `summoning_circle` | A summoning circle: a violet ring on the ground with arcs of energy leaping up from it and a portal's light (procedural) | ring, arc, lamp | 420 x 360, base | loop | intensity (0.9) | Good: lively arcs over the ring. |
| `dragon_breath` (new) | Dragon breath: a long, wide gout of violet fire from the anchor (the jet's flame recoloured: it moves as the orange one); it blows to the right | jet; look drake, lamp | 640 x 340, left | loop | intensity (0.95), aim (5) | Good: a long violet gout of fire (the jet, recoloured). |
| `wisp` (new) | A wisp: a small pale ball of cyan light that bobs and drifts about its place (the orb model, small) | orb; lamp | 120 x 120, centre | loop | intensity (0.3) | Fair: a small cyan ball that bobs about; it reads as a wisp in motion, and only in the orb's cyan. |
| `magic_vortex` (new) | A magic vortex: a swaying column of violet fire rising from the ground (the vortex model recoloured), motes circling it | whirlwind; look arcane, glow, lamp | 320 x 480, base | loop | intensity (0.8), sway (0.5) | Fair: a swaying violet column (since study L's vortex one column where it often split in two). |

### Ambient (5)

| effect | what it is | plays | box, anchor | runs | parameters (default) | how it looks |
|---|---|---|---|---|---|---|
| `heat_haze` | Heat haze: shimmering air over hot ground or an engine, the fire model's heat with nothing drawn | fire; look invisible | 320 x 320, base | loop | intensity (0.6) | Works in motion only: a still shows nothing, the shimmer is subtle, and the invisible fire's heat lights the scene a little. |
| `fireflies` | Fireflies: small yellow-green lights wandering and blinking in the dark (procedural) | glow | 360 x 240, centre | loop | count (16) | Good, and small. |
| `drifting_embers` | Drifting embers: embers rising on warm air and drifting away, as over a burnt-out fire (carried by the flow of an invisible fire) | fire; look invisible, embers | 360 x 360, base | loop | density (1) | Faint: a few embers carried up by an invisible fire's flow; reads in motion, hardly in a still. |
| `smouldering_ground` | Smouldering ground: thin smoke and a dull red glow where a fire has burnt out | smoke; look smoke, lamp, burst | 420 x 300, base | loop | intensity (0.18) | Fair: a thin wisp and a dull red glow; the wisp rises higher than smoulder smoke would. |
| `dust_motes` | Dust motes: specks of dust drifting slowly through a beam of light (procedural) | glow, lamp | 300 x 300, centre | loop | count (24) | Fair: faint by design; needs a dark scene. |

### Sparks and debris (8)

| effect | what it is | plays | box, anchor | runs | parameters (default) | how it looks |
|---|---|---|---|---|---|---|
| `sparks` | A fountain of sparks thrown up from a point, falling back (a fuse, a broken cable; procedural) | sparks, lamp | 200 x 220, base | loop | count (8) | Fair: a small fountain of the fireball's short fuse sparks. |
| `welding_sparks` | Welding: a blue-white flickering arc light with sparks spraying from it (procedural) | glow, sparks, lamp | 220 x 200, base | loop | count (14) | Good: a flickering arc light and sparks. |
| `grinder_sparks` | A grinder's sparks: a fast spray of sparks thrown sideways from a point 40 world pixels above the anchor (procedural) | glow, burst, lamp | 360 x 200, base | loop | density (0.8), direction (1) | Fair: a sideways spray of short sparks; it replaced a spark shower whose sparks died before they fell far. |
| `debris_shower` | A shower of debris: rocks and burning fragments thrown up and falling back, with a puff of dust | smoke; look dust, ceiling, burst, shock | 400 x 300, base | once, 3 s | power (1) | Good: debris and embers thrown up, with a puff of dust from the smoke model. |
| `electric_arc` | An electric arc jumping between two points 200 world pixels apart, flickering, with sparks (procedural) | arc, sparks, lamp | 260 x 120, base | loop | intensity (1) | Good: a lively flickering arc. |
| `lightning_strike` | A lightning strike: a branching bolt from the sky to its anchor, struck three times, a flash and a scorch (procedural) | arc, flash, lamp, scorch, shock, burst | 360 x 700, base | once, 1.2 s | power (1) | Good: a branching bolt struck three times, a flash and a scorch. |
| `ember_burst` | An ember burst: a log collapsing in a fire, a puff of embers thrown up that drift and cool (procedural) | burst, lamp | 360 x 360, base | once, 3.5 s | power (1) | Good: a puff of embers that drift and cool. |
| `ball_lightning` (new) | Ball lightning: a crackling ball of plasma drifting about, throwing short arcs (the orb model with procedural arcs) | orb; arc, lamp | 200 x 200, centre | loop | intensity (0.5) | Good: a boiling ball with arcs crackling off it. |

### Light (7)

| effect | what it is | plays | box, anchor | runs | parameters (default) | how it looks |
|---|---|---|---|---|---|---|
| `muzzle_flash` | A muzzle flash: a white-hot core, a long spike along the barrel and short side spikes, sparks, a flash of light (procedural) | glow, arc, lamp, burst | 160 x 100, centre | once, 0.12 s | direction (1) | Good; it lasts 0.12 s. |
| `signal_flare` | A burning signal flare on the ground: a red glare that flickers, sparks and a trail of pink smoke | smoke; look signal, glow, sparks, lamp | 200 x 320, base | loop | intensity (1) | Good: a red glare and pink smoke. |
| `lantern` | A lantern: a steady warm point of light with a soft halo and a gentle flicker (procedural) | glow, lamp | 120 x 120, centre | loop | intensity (1) | Good: a warm point with a soft halo. |
| `flashbang` | A flashbang: a white flash that blinds the whole scene for a moment, a bang (a shock front) and sparks (procedural) | flash, glow, shake, shock, burst | 400 x 300, centre | once, 1.5 s | power (1) | Good: the whole picture whites out, as it should. |
| `shockwave_ring` | A shockwave ring: a glowing ring racing out along the ground and a shock front bending the air (procedural) | ring, shock, scorch | 600 x 260, base | once, 1.2 s | power (1) | Good: the ring and the bending air read well. |
| `plasma_orb` (new) | A plasma orb: a ball of cyan plasma that boils inside a soft edge and breathes, lighting what is near in cyan | orb | 140 x 140, centre | loop | intensity (0.5), pulse (0.4) | Good: a boiling cyan ball that lights what is near it. |
| `energy_core` (new) | An energy core: a bright ball of plasma that throbs hard, inside a ring of light | orb; ring, lamp | 200 x 200, centre | loop | intensity (0.85) | Good: a bright ball that throbs hard inside a ring of light. |

### Dust (5)

| effect | what it is | plays | box, anchor | runs | parameters (default) | how it looks |
|---|---|---|---|---|---|---|
| `dust_impact` (new) | A dust impact: dust thrown up where something heavy hits the ground, rolling out low both ways and settling | dust; look dirt, burst | 480 x 200, base | once, 3.5 s | power (0.85) | Good: the dust model's burst rolls out low and settles; the finest detail of the new entries. |
| `landing_dust` (new) | Landing dust: a small puff of dust where someone lands, running out low both ways | dust; look dirt | 240 x 80, base | once, 2.5 s | power (0.4) | Good: small and quick. |
| `ground_slam` (new) | A ground slam: a heavy blow on the ground: a shock front, a ring racing out, flying rocks, a camera shake and a wide wall of dust | dust; look dirt, ring, shake, shock, burst | 680 x 260, base | once, 3.5 s | power (1) | Good: the ring, the shake, the rocks and a wide wall of dust read as one heavy blow. |
| `sand_burst` (new) | A sand burst: a shell or a heavy blow in sand throws up a tall spray of pale sand that rolls out and settles | dust; look sand, burst | 560 x 280, base | once, 3.5 s | power (1) | Good: a tall pale spray that rolls out; the rocks thrown are the compositor's dark debris, not sand. |
| `dust_devil` (new) | A dust devil: a column of dust rising from hot ground, swaying and spreading towards its top (the vortex model) | whirlwind; look devil_dust | 300 x 420, base | loop | intensity (0.65), sway (0.6) | Fair: the column sways and spreads at its top into a haze; its intensity is kept at 0.5 or more, as the model can lose a thin column (less often since study L, not never). |

### Weather (6)

| effect | what it is | plays | box, anchor | runs | parameters (default) | how it looks |
|---|---|---|---|---|---|---|
| `ground_fog` (new) | Ground fog: a low white bank of fog lying along the ground and rolling slowly, four releases of the fog model | fog; look fogbank | 900 x 120, base | loop | intensity (0.7), wind (0.55) | Good: a continuous low bank that rolls slowly; its four releases still show as gentle humps. |
| `frost_mist` (new) | Frost mist: a thin, pale blue mist creeping over frozen ground, calm and slow | fog; look rime | 560 x 100, base | loop | intensity (0.5) | Fair: pale and thin by design; it needs a dark ground to read. |
| `swamp_mist` (new) | Swamp mist: a murky grey-green mist hanging over still water, with a few faint marsh lights drifting in it | fog; look marsh, glow | 760 x 140, base | loop | intensity (0.6) | Fair: murky and faint, and its marsh lights are small. |
| `sky_cloud` (new) | A cloud: a white mass that boils slowly and drifts, lit by the moon and the scene | cloud; look fleece | 520 x 260, centre | loop | wind (0.6), intensity (0.8) | Good: a soft white cloud that boils slowly (drawn by a look: the cloud model's own renderer gave colour fringes). |
| `storm_cloud` (new) | A storm cloud: a dark bank of cloud drifting, lit from inside by lightning every few seconds, a bolt below it | cloud; look storm, lamp, arc, flash | 1100 x 360, centre | loop | wind (0.62), every (5) | Fair: a flash lights the bank from inside and the bolt reads, but between flashes the bank is dark and thin. |
| `dust_storm` (new) | A dust storm: a wall of brown dust rolling past, puffs forming upwind and thinning downwind (the cloud model) | cloud; look grit | 1300 x 300, base | loop | intensity (0.95), wind (0.85) | Fair: a low brown band rolling past, not a towering wall. |

## 2. The format

A catalogue is written in the scene-script language ([COMPOSE.md](COMPOSE.md) §4), with two more statements. Each
file declares, before its first entry, the learned effects and the field-shader looks its entries share; then entries,
each an `entry` line, its `param` lines and the statements of a small scene:

```
effect fire = "fire.nvfx"                                    # once per catalogue (models.nvfxl)
look gas = like flame, hue 225, emission 2.8, soot_density 0.4

entry blue_gas_flame, category fire, size 120 x 190, anchor base, loop, show 2,
      about "A blue gas flame: the fire model, calm, recoloured blue by a field-shader look, with a blue light"
param intensity = 0.55, range (0.2, 1), about "how big it burns"
module flame = fire, size 64, width 110 * scale, at (X, Y), sink 0.05, feather 0.15, look gas, start 0, seed 81,
               controls (0.55, 0.5, 0.2), intensity intensity, glow (0.3, 0.55, 1.6)
field cool = lamp, at (X, Y - 20 * scale), radius 50 * scale, strength 0.6 * intensity, colour (0.35, 0.55, 1)
```

| statement | keys |
|---|---|
| `entry NAME` | `category C` (fire, smoke, explosions, steam, magic, ambient, sparks, light, dust, weather, or a new one), `size W x H` (its box at scale 1, world pixels), `anchor base`, `centre` or `left` (the middle of the box's left edge: a jet's nozzle), `loop` (plays until stopped) or `once S` (a one-shot of S seconds), `show S` (a moment that shows it, seconds after the spawn; the gallery's), `about "..."` (one line) |
| `param NAME = DEFAULT` | `range (LO, HI)` (values set are clamped to it), `about "..."` |

In an entry, `t` is the time since the spawn, `X` and `Y` the anchor, `scale` the scale it was spawned at, `variant`
a number in [0, 1) that differs from spawn to spawn, and `alive` 1 while it plays (0 once stopped or ended). A
parameter is a value by its name. An entry may declare modules (single ones: tiles stay where they are declared),
fields, emitters, `let`s, its own looks, rules (`at T:` and `when ...:`, which fire once per spawn, or every time
with `repeat`), `every frame:` couplings, a flash (`light flash (r, g, b)`) and a camera shake (`camera x ..., y ...`).
The stage owns the picture, the clock, the bus, the particles and the camera's place; an entry cannot set them, read
another instance, or wait for a shock front by number (fronts are counted over the whole stage). The parser and checker
give every error as `file.nvfxl:line:column: message`.

**Why the scene-script language.** An entry is a small composed scene: modules of a model at a size and place, with
their controls tied to parameters, fields that act on them, emitters, and rules over time (the grenade's flash at 0,
its shock front, its scorch mark fading). The scene language already says all of that, and it already has the parser,
the checker with line and column, the runtime that plays without allocating and gives the same frames on any number of
threads, the overlap of drawing with the next frame, the field bus that couples effects, and the tools
(`nvfx_scene_script`, the viewer). A table format (JSON, TOML, a C++ registry) would have needed a second language for
the behaviour or a second runtime; it would not have let an entry be anything a scene can be. With the scene language,
an entry is checked and played by the same code as every scene, and a stage is itself a scene script
(`nvfx_stage_script`, `nvfx_gallery --print-stage`) that the scene tools check and play like any other (its rooms
stay empty until their inputs are set).

**What a stage does to an entry.** A stage is one scene with *rooms*: for each effect it hosts, a number of instances
that may play at once. Each room gets a copy of its entry, renamed (`s3_flame`, `s3__X`, `s3__intensity`), whose
anchor, scale, variant, parameters, spawn time and end time are the scene's **inputs**. A spawn sets them; nothing is
built when a game spawns, moves or stops an effect, so a stage plays without allocating after it is created. The copy
is rewritten so that:
- `t` is the time since the spawn, and the entry's rules fire once per spawn (they become `repeat` rules guarded by
  "not fired since this spawn");
- its learned modules start at stage creation, asleep (`waiting`: their warm-up happens then, not in a frame); the
  first spawn wakes them, later spawns restart them from their start point with a seed that changes with `variant`;
- a stop, or the end of a one-shot, stops its modules, gates its emitters, fields, flashes and shakes with `alive`, and
  frees its room (with `steal`, a spawn when every room is busy takes the oldest's);
- looping modules play in **shards** of 8 seconds (`shards`, [COMPOSE.md](COMPOSE.md) §4.3): a fresh rollout starts
  ahead of time and takes over in a half-second blend, so a fire or a smoke plume never drifts or dies however long it
  plays; a shard that fades early is replaced at once;
- flashes and camera shakes of all instances add up.

`nvfx_gallery --library library --print NAME` prints an entry alone as the stage plays it.

### 2.1 Procedural elements added for the library

All in the compositor (`src/compose`), allocation-free (pre-sized pools), deterministic, and tested
(`tests/test_library.cpp`). None is learned: they are drawn, and say so in the entries that use them.

| element | what it is | used by |
|---|---|---|
| `emit arc` | a lightning bolt or an electric arc between two points: midpoint displacement with forks, a core and a halo, a new shape every frame or `rate` times a second | `lightning_strike`, `electric_arc`, `dark_portal`, `summoning_circle`, `teleport_flash`, `muzzle_flash` |
| `emit ring` | a glowing ring, squashed into an ellipse on the ground | `shockwave_ring`, `healing_circle`, `summoning_circle`, `magic_burst`, `teleport_flash` |
| `emit glow` | a soft disc of light, or motes wandering and flickering | `lantern`, `fireflies`, `dust_motes`, `witch_fire`, `welding_sparks`, `portal`, ... |
| `field lamp` | a light that is not on the bus: lights smoke and steam around it in its colour | most entries (the warm light of a fire, a portal's violet) |
| `hue` (look) | the field shader's emitted colours turned about the grey axis: recolours | `blue_gas_flame`, `witch_fire`, `magic_burst`, `dark_portal` |
| `burst ... sparks` | short fast bright embers thrown within a cone | `muzzle_flash`, `grinder_sparks`, `lightning_strike`, `flashbang` |

## 3. The API

`include/neuralfx/nvfx_library.h`, in `libnvfx` (C, as the scene API, [ENGINES.md](ENGINES.md) §7):

```c
nvfx_library_desc ld;
nvfx_library_desc_init(&ld);
ld.catalogue_dir = "library/";                         /* every *.nvfxl in it */
ld.models_dir = "effects/";                            /* fire.nvfx, smoke.nvfx, ... vortex.nvfx */
nvfx_library* lib;
nvfx_scene_error err;
if (nvfx_library_open(&ld, &lib, &err) != NVFX_OK) printf("%s\n", err.message);   /* "fire.nvfxl:12:7: ..." */

nvfx_stage_room rooms[] = {{"campfire", 4, 0}, {"grenade", 8, 1}};   /* up to 4 campfires, 8 grenades (stealing) */
nvfx_stage_desc sd;
nvfx_stage_desc_init(&sd);                             /* 960 x 540, 30 fps, 2 threads */
sd.rooms = rooms;
sd.n_rooms = 2;
nvfx_stage* stage;
nvfx_stage_create(lib, &sd, &stage, &err);             /* loads the models, warms them up: the only allocations */
nvfx_scene* scene = nvfx_stage_scene(stage);

nvfx_fx fire;
nvfx_stage_spawn(stage, "campfire", 300, 470, 1.0f, -1.0, &fire);  /* anchor at (300, 470), scale 1, now */
nvfx_stage_set_param(stage, fire, "intensity", 0.4f);
/* every frame: */ nvfx_scene_step(scene, dt, NULL); nvfx_scene_render(scene, rgba, stride);
nvfx_stage_stop(stage, fire, -1.0);
```

- **Listing**: `nvfx_library_count`, `_find`, `_get_effect` (name, category, description, the models and procedural
  elements it uses, its box, anchor, loop or duration, a moment to show it), `_get_param` (name, description,
  default, range), `_category_count` and `_category`, `_list_models` (the model files an effect needs).
- **Stages**: `nvfx_stage_create` builds the scene with the rooms asked for and loads the models they need;
  `nvfx_stage_scene` is an ordinary `nvfx_scene` (step, render, sample fields, overlap). `nvfx_stage_spawn` (position,
  scale, time; `NVFX_ERROR_MEMORY` when the effect has no free room and no `steal`), `_set_param` and `_get_param`
  (clamped to the range), `_move`, `_stop` (now or at a time; a pending spawn is cancelled), `_get_fx` (waiting,
  playing, ended; where, its scale, its age), `_free_rooms`, `_set_camera`, `_clear`. A handle is a room and a
  generation: it goes stale when its room is used again.
- **No allocation per frame**, and none in spawn, move, parameters or stop (they set the scene's inputs):
  `nvfx_alloc_test` counts them (a stage with 21 spawns over 149 frames, with and without overlap: 0), as does
  `nvfx_gallery --check` for every entry.
- **Deterministic**: the same calls at the same times give the same frames, on any number of threads and with or
  without overlap (`Library.TheSameCallsGiveTheSameFramesOnAnyThreads`). An instance's random choices come from its
  room's seeds and its `variant` (from the spawn count), not from the clock.
- **Threads**: as scenes. A library is read-only once opened and may be shared; a stage is used from one thread at a
  time and runs `threads` workers of its own.
- **Errors**: a catalogue error gives the file, line and column; an error found only when the models are loaded (a
  control name the model does not have) is reported against the catalogue, not the generated stage
  (`bad.nvfxl:3:47: ...`).

`examples/library_host.c` is the same in plain C (built as `nvfx_library_host`): given a catalogue and the models'
folder it lists the catalogue and plays a stage of a few effects (spawned, a parameter set, one moved, one stopped, the
cost per frame and the heat under each), and with `--self-test` it checks the API without data (listing, a catalogue
error, spawns, no room, steal, stop, rooms freed, camera, clear, the same frames on 1 and 3 threads).

## 4. The gallery

The models' folder, `$NEURALVFX_DATA/library/models` by default, holds the eleven learned effects under the names
`models.nvfxl` gives them: the v2 fire, smoke and explosion, study D's steam, round 5's fog, dust, cloud and jet, and
study L's magic, orb and vortex ([REPORT.md](REPORT.md) §12.1, which replaced study D's magic and round 5's orb and
vortex). Links are enough:

```sh
mkdir -p $NEURALVFX_DATA/library/models && cd $NEURALVFX_DATA/library/models
ln -s ../../v2/models/fire.nvfx ../../v2/models/smoke.nvfx ../../v2/models/explosion.nvfx .
ln -s ../../fx/models/d/steam.nvfx .
ln -s ../../fx_atmos/models/d/fog.nvfx ../../fx_atmos/models/d/dust.nvfx ../../fx_atmos/models/d/cloud.nvfx .
ln -s ../../fx_energy/models/d/jet.nvfx .
ln -s ../../l_swirl/models/v3/magic.nvfx ../../l_swirl/models/v3/orb.nvfx ../../l_swirl/models/v3/vortex.nvfx .
```

`tools/nvfx_gallery.cpp` plays every entry through the C API, each alone on a stage fitted to its box:

```
nvfx_gallery --library library --models DATA/library/models --sheet sheet.png --video walk.mp4 --threads 2
nvfx_gallery --library library --models DATA/library/models --video new.mp4 --clip 2.4 --only a,b,c   # 2.4 s each
nvfx_gallery --library library --models DATA/library/models --keyframes DIR --verify results/library/frozen.csv --only a,b
nvfx_gallery --library library [--models DIR] --check [--require-models]
nvfx_gallery --library library --list [--markdown]        # the catalogue (the table above comes from it)
nvfx_gallery --library library --print NAME               # an entry alone, as the stage plays it
nvfx_gallery --starts DATA/library/models/fire.nvfx       # a model's start points and their controls
nvfx_gallery --library library --models DIR --shot 960x540 --ground 470 --spawn campfire@220,470,1,0.3 ... --frames 45 --png f.png
```

- `--sheet`: a labelled contact sheet (name, category, one-shot or loop), each cell at the entry's `show` moment.
- `--video`: a walk through every entry (or those of `--only`), 4 seconds each (one-shots: their length; `--clip S`:
  S seconds each), with its name.
- `--check`: the catalogue parsed and every entry checked as a stage of its own; every entry whose models are present
  (procedural ones always) played as two instances, its parameters set to both ends of their ranges, one moved, one
  stopped and spawned again, heap allocations counted in the frame loop and in those calls (0 for each), and something
  drawn.
- `--keyframes` and `--verify`: each entry's `show` frame as PNG, compared by SHA-256 with frozen rows
  (`library_keyframe,NAME,file,bytes,sha256`; `results/library/frozen.csv` also records the SHA-256 of the eleven model
  files they were made with).
- `--shot`: a stage played as a game would (frame 0 drawn, the spawns, then N frames): the Godot test's reference.

Outputs go under `$NEURALVFX_DATA/library` (the sheet, the video, keyframes); only the two sheets in `docs/figures`
(`library_sheet.png`, `library_sheet_new.png`, scaled down) are in git.

**Tests** (`ctest`): `neuralfx.LibraryCatalogueCheck` (no data: the catalogue and every procedural entry);
`neuralfx.LibraryHostExample` (the C example's self-test); `neuralfx.LibraryCataloguePlays` (every entry, with the
eleven models in `NEURALFX_MODELS_LIBRARY`, by default `$NEURALVFX_DATA/library/models`; skipped without them);
`neuralfx.LibraryKeyframesFrozen` (the keyframes of `campfire`, `blue_gas_flame`, `grenade`, `steam_vent`, `portal`,
`lightning_strike`, `ground_fog`, `dust_impact`, `flamethrower`, `plasma_orb` and `dust_devil` against
`results/library/frozen.csv`; skipped without the models); the `Library.*`, `Glows.*`, `Lamps.*` unit tests and the
library stage in `nvfx_alloc_test`.

## 5. Godot

`NeuralVFXStage` (engines/godot, [ENGINES.md](ENGINES.md) §8) is a `NeuralVFXScene` that plays a stage: set
`library_dir` (the catalogue), `effects_dir` (the models), `stage_size`, `ground` and `rooms` (a Dictionary of effect
name to count, or to `[count, steal]`), then:

```gdscript
var stage := NeuralVFXStage.new()
stage.library_dir = "res://library"
stage.effects_dir = "res://effects"
stage.rooms = {"campfire": 2, "grenade": [4, true]}
add_child(stage)
print(stage.list_effects(), stage.get_categories(), stage.get_effect_info("campfire"))
var fx := stage.spawn("campfire", Vector2(300, 470))      # scale 1, now; 0 when there is no room
stage.set_param(fx, "intensity", 0.4)
stage.move_fx(fx, Vector2(320, 470))
stage.stop(fx)
```

`demo/library.tscn` (`library.gd`) is the same in a window: left and right choose one of 22 effects, a click spawns it
under the mouse, a right click stops the last. The headless test (`engines/godot/test.sh`, step 5) has `nvfx_gallery
--shot` draw a stage with five library effects (a campfire, a steam vent, a magic burst, an electric arc and a plasma
orb) spawned at given times, and `demo/test_library.gd` play the same stage in a `NeuralVFXStage` in Godot 4.4.1: the
listing, an effect's description and a jet's anchor, the spawns, a full room and an unknown effect refused, the frame
the same to the byte as `nvfx_gallery`'s, heat over the campfire, a parameter, a move, a stop, a stale handle.

## 6. Adding effects

**A new trained model**: one line in `library/models.nvfxl` and entries that play it, in a new or existing file; no
code, no rebuild. This is how fog, dust, cloud, jet, orb and vortex joined (23 entries; the jet also needed an anchor
at the left edge of the box, below); for example the dust impact:

```
effect dust = "dust.nvfx"                                     # models.nvfxl; the file goes into the models folder
look dirt = shader, emission 0, soot_density 2.2, soot_albedo 0.6, sky 0.45, shadow 0.4, scene_light 1.2, relief 3,
            tint (0.8, 0.62, 0.42)                            # (the moor scene's look: dust has no heat to draw)

entry dust_impact, category dust, size 480 x 200, anchor base, once 3.5, show 0.8,
      about "A dust impact: dust thrown up where something heavy hits the ground, rolling out low both ways and settling"
param power = 0.85, range (0.3, 1), about "how hard the hit"
module heap = dust, size 128, width 480 * scale, at (X, Y), sink 0.03, feather 0.1, look dirt, start 9, seed 601,
              controls (0.85, 0.5, 0.4), intensity power, opacity 1 - smooth((t - 2.4) / 1)
at 0:
  burst at (X, Y - 4 * scale), radius 8 * scale, embers 0, debris 12, speed 260 * scale * power
```

Then:
1. `nvfx_gallery --library library --check` (the catalogue, without the model), and with `--models` (it plays);
2. pick start points by their controls (`nvfx_gallery --starts DATA/library/models/fog.nvfx` lists them): a module's
   `start N` should have controls near the ones it plays at (a start point of a weak fire played at full intensity
   grows slowly and looks young, one of a strong side wind leans for its first seconds); the shards pick the nearest
   start point by controls for every later shard anyway;
3. look at it: `nvfx_gallery --sheet ... --only dust_impact` and `--video`, and play loops for half a minute or more
   (`--shot ... --frames 900`); adjust the size, the look, the parameters' ranges (a range that reaches controls the
   model never saw in training gives what it gives);
4. if the entry should be guarded against changes, add it to `--only` of `neuralfx.LibraryKeyframesFrozen` and its
   row to `results/library/frozen.csv` (`nvfx_gallery --keyframes DIR --only NAME` writes the rows to copy into
   `DIR/keyframes.csv`).

A model with other control names is used by them (`spin 0.8` for magic, `aim`, `pulse` and `sway` for the jet, the
orb and the vortex). A new category is just a new name in `category` (`dust` and `weather` came with round 5).

**What adding six models taught** (no step of the recipe changed; these are things to know):
- **Read the model's document first**: where its material sits in the tile decides where the module stands. Fog lies
  in the bottom quarter of its tile and spreads along it, so ground fog is several narrow releases side by side (one
  wide release stands up like a cumulus); the cloud's mass sits in the middle of its tile, so a smoke screen's tiles
  stand half below the ground; the orb is a ball in the middle of its tile; the jet's nozzle is 5% in from the left
  edge and 40% up.
- **A model that blows one way needs its own anchor.** The jet always blows to the right, as fire always rises (a tile
  cannot be mirrored), so its entries are `anchor left`: the spawn point is the nozzle and the box lies to its right.
  Their `aim` parameter is in degrees (-20 to 30) and mapped onto the control (`aim (aim + 20) / 50`).
- **Models without heat need a look.** Fog, dust and cloud carry soot only; their entries draw them with field-shader
  looks lit by the scene (the moor scene's frost, storm and dirt; at night a denser one than the moor's for fog). The
  cloud model's own renderer gave colour fringes, so even the white cloud has a look.
- **A one-shot model plays as the explosion does**: `once`, and its module faded out by `opacity` before the end.
- **Names**: `vortex` is a field of the language, so the vortex model is `effect whirlwind = "vortex.nvfx"`; locals such
  as `core` and `by` are words of the language too, and the checker says so with the line and column.
- **The start point by controls matters most for the control you can see.** A rocket exhaust at low turbulence
  started its later shards from the start point of a jet aimed down (the nearest by all three controls); a turbulence
  of 0.45 made the level one nearest.
- **Keep a model inside the range where it holds up.** The vortex can lose a thin column at low intensity
  ([EFFECTS_ENERGY.md](EFFECTS_ENERGY.md) §4.4), so its entries' intensity ranges start at 0.5. Fog lifted high stays up
  as pale plumes ([EFFECTS_ATMOSPHERE.md](EFFECTS_ATMOSPHERE.md) §6), so no entry pushes fog.

**A new look** (a recolour) is a `look` line before the first entry of a file (shared) or inside an entry (its own).
It changes colours, not motion; say so in the entry's description.

## 7. Costs and limits

- **A stage's cost is its picture and its playing instances.** A sleeping room costs memory only (its modules are
  made and warmed up, but neither step nor draw). In the gallery, on one thread of a busy shared machine, drawing the
  stage itself took about 2.5 ms at 320 x 240, growing with its area (about 30 ms at 1120 x 840); on top of that a fire
  flame of 128 pixels took about 3 ms, one of 64 pixels about 1 ms, a smoke or steam tile of 256 pixels about 15 ms,
  and the procedural elements little. `nvfx_gallery` prints each entry's time per frame.
- **Spawning a learned effect again costs a warm-up**: a later spawn restarts its modules from a start point, which
  runs that start point's warm-up in the frame (16 to 20 ms for a fire or steam module of 128 pixels, more for 256).
  The first spawn of a room does not (its modules warmed up at stage creation). The shards of looping modules each
  warm up a fresh rollout every 8 seconds the same way, spread over the frames by their offsets.
- **Scale**: learned modules keep their pixels; far above scale 1 they blur, far below they waste work. The
  procedural elements scale cleanly.
- **The picture is the stage's**: a stage draws its own background (the scene's night sky and ground) into an opaque
  frame; a game composites it as it composites a scene (ENGINES.md). Effects light each other and the scene's smoke
  through the bus, but not the game's own geometry.
- **Thin flames**: the fire model makes one flame per tile, so wide fires are several flames side by side; seen
  close they read as separate tongues.
- **One-shot explosions** use the explosion model's medium start points; the strongest fill their tile and square off
  at its edges.
- **Jets blow to the right only** (the model's direction; a tile cannot be mirrored), between 20 degrees down and 30
  up. A game that needs a flame to the left has none yet.
- **The vortex** can lose its column at low intensity (less often since study L's vortex, not never), so its entries
  stay at 0.5 or above; the orb's ball follows its runs no better than a coarse simulation, even after study L
  (REPORT §12.1), but both play endlessly without trouble, which is what a library entry does.
- **Fog stays where it is put**: nothing in the library pushes it, as fog lifted by a blast stays up as pale plumes.
  Fog, dust and cloud give no light and no heat haze.

## 8. Judged by eye: what changed

Every entry was looked at in the contact sheet and the walk-through video, and the bad ones changed or replaced:
- **Long loops** drifted: a smoke plume broke up after about half a minute, and some seeds of fire and steam died out
  within seconds. Looping modules now play in shards of 8 seconds that blend into each other, each starting from the
  start point nearest the module's controls, and a shard that fades early is replaced at once (`shards`,
  [COMPOSE.md](COMPOSE.md) §4.3).
- **Explosions** from the strongest start points fill their tile and show its square edge; the one-shots use the
  medium ones.
- **Smoke above fires** (`bonfire`, `burning_wreck`) floated as a separate cloud above the flames; their smoke tiles
  now stand in the flames, so the smoke rises out of them.
- **Boxes the tiles could not fill** (`smoke_column`, `black_oil_smoke`): two side-by-side tiles 360 pixels wide under a
  700-pixel box made a short, thin plume; one larger tile (600 to 640 pixels) fills the box.
- **`brazier`** started its main flame from a start point of strong turbulence and side wind, which made a thin red
  streak for its first seconds; it now starts from the calm one.
- **`steam_vent`** was hard to see (the vapour look too dark) and one seed lost its jet; a brighter look, another seed
  and the shards' healing fixed both.
- **Replaced**: `waterfall_mist` (a ceiling over five steam plumes made a flat slab, not a veil: the plumes are now
  `hot_springs`, which is what they look like); `spark_shower` (sparks pouring down from a point died before they had
  fallen far: now `grinder_sparks`, a sideways spray); `drifting_embers` (procedural embers fell straight back to the
  ground: they now ride the flow of an invisible fire, so they rise).
- **Kept although weak**: the faint `drifting_embers` and `heat_haze`, which read in motion only.

The six newer models' entries (round 5) were judged the same way, from sheets, a 30-second play of every loop and a
walk-through video:
- **`poison_smoke` was replaced by `poison_gas`**: the smoke model always rises, and held down by a ceiling it spread
  into a marbled mushroom; the fog model is a heavy gas that stays on the ground.
- **Fog at night** was nearly invisible with the moor's look (the moor's ground is lighter): the fog entries use a
  denser look, releases closer together and of different sizes, so they join into one bank.
- **The smoke screen** stood above the ground until its tiles were set half below it (the cloud's mass is in the
  middle of its tile); **the dust storm** became three large tiles low down, from three small floating puffs.
- **The white cloud** is drawn by a look: the cloud model's own renderer gave pink and green fringes.
- **The rocket thruster** lost its exhaust after half a minute: its later shards started from a jet aimed down (see
  §6); a higher turbulence fixed it.
- **The fire whirl** glowed as a large orange square (the warm air at the top of its tile drawn as flame); a dimmer
  look keeps the column and a softer haze.
