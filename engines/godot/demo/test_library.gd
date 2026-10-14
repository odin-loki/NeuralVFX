# The headless test of the effect library in Godot (engines/godot/test.sh): a NeuralVFXStage lists the catalogue,
# spawns five effects (learned and procedural) at given times, plays 45 frames and saves the frame as PNG. Checks: the
# catalogue's listing and an effect's description; spawns, an unknown effect and a full room refused; given
# nvfx_gallery's frame of the same stage and spawns (--reference), the same pixels to the byte; heat over the campfire;
# the instances' states, a stop and a stale handle. Exit code: the number of failed checks.
#
#   godot --headless --path engines/godot/demo --script res://test_library.gd -- --library DIR --library_models DIR
#         --out PNG --frames 45 [--reference library_frame.png]
#
# The spawns are those test.sh gives nvfx_gallery --shot (keep them the same).
extends SceneTree

const Paths = preload("res://paths.gd")

var failures := 0


func check(ok: bool, what: String) -> void:
	print(("  ok    " if ok else "  FAIL  ") + what)
	if not ok:
		failures += 1


func _initialize() -> void:
	var args := Paths.args()
	print("NeuralVFX library test: %s, effects in %s" % [args.library, args.library_models])
	var stage := NeuralVFXStage.new()
	stage.library_dir = args.library
	stage.effects_dir = args.library_models
	stage.stage_size = Vector2i(960, 540)
	stage.ground = 470
	stage.rooms = {"campfire": 1, "steam_vent": 1, "magic_burst": 1, "electric_arc": 1, "plasma_orb": 1}
	stage.threads = 2
	stage.precision = 1  # float, as the reference
	stage.playing = false  # the frames are stepped here
	root.add_child(stage)
	stage.load_scene()
	if not stage.is_loaded():
		printerr("cannot load the stage: ", stage.get_error())
		quit(1)
		return

	var names := stage.list_effects()
	print("  %d effects in %s" % [names.size(), stage.get_categories()])
	check(names.size() >= 30 and names.has("campfire") and names.has("lightning_strike"), "the catalogue lists its effects")
	var info := stage.get_effect_info("campfire")
	print("  campfire: ", info)
	check(info.category == "fire" and info.looping and info.params.size() == 2 and info.params[0].name == "intensity", "an effect's description")
	var jet := stage.get_effect_info("flamethrower")
	check(jet.anchor == "left" and jet.models == "jet", "a jet is anchored at its nozzle, on the left of its box")

	var fire: int = stage.spawn("campfire", Vector2(220, 470), 1.0, 0.3)
	var vent: int = stage.spawn("steam_vent", Vector2(480, 470), 0.8, 0.4)
	var burst: int = stage.spawn("magic_burst", Vector2(760, 260), 1.0, 0.6)
	var arc: int = stage.spawn("electric_arc", Vector2(620, 470), 1.0, 0.2)
	var orb: int = stage.spawn("plasma_orb", Vector2(110, 150), 0.8, 0.5)
	check(fire != 0 and vent != 0 and burst != 0 and arc != 0 and orb != 0, "five spawns")
	check(stage.spawn("campfire", Vector2(300, 470)) == 0, "no second campfire: its one room is taken")
	check(stage.spawn("no_such_effect", Vector2(300, 470)) == 0, "an unknown effect is refused")
	check(stage.get_fx_info(fire).state == "waiting", "the campfire waits for its time")

	var frames: int = args.frames
	for f in range(frames):
		stage.advance(1.0 / stage.get_fps())
	print("  %d frames, frame %d at t %.2f s" % [frames, stage.get_frame(), stage.get_time()])

	var img: Image = stage.get_image()
	check(img != null and img.get_width() == 960 and img.get_height() == 540, "the frame is 960 x 540")
	if img != null:
		check(img.save_png(args.out) == OK, "saved " + args.out)
		if not str(args.reference).is_empty():
			var ref := Image.load_from_file(args.reference)
			if ref == null:
				check(false, "read the reference " + args.reference)
			else:
				ref.convert(Image.FORMAT_RGBA8)
				check(ref.get_data() == img.get_data(), "the same pixels as nvfx_gallery's frame (" + args.reference.get_file() + ")")
	var heat := stage.get_field_max(NeuralVFXScene.FIELD_HEAT, Rect2(180, 300, 80, 170))
	print("  heat over the campfire (max) %.2f" % heat)
	check(heat > 0.1, "heat where the campfire burns")
	var fi := stage.get_fx_info(fire)
	print("  the campfire: ", fi)
	check(fi.state == "playing" and fi.effect == "campfire" and absf(fi.spawned - 0.3) < 0.02, "the campfire plays since its time")
	check(stage.get_fx_info(burst).state == "playing", "the magic burst plays (0.9 s of its 2.2)")
	check(stage.set_param(fire, "intensity", 0.4) and is_equal_approx(stage.get_param(fire, "intensity"), 0.4), "a parameter set")
	check(stage.move_fx(vent, Vector2(500, 470)), "an instance moved")
	check(stage.stop(fire), "the campfire stopped")
	stage.advance(1.0 / stage.get_fps())
	check(stage.get_fx_info(fire).state == "ended" and stage.free_rooms("campfire") == 1, "it ended and its room is free")
	var again: int = stage.spawn("campfire", Vector2(240, 470))
	check(again != 0 and stage.get_fx_info(fire).state == "none", "spawned again in its room: the old handle is stale")
	check(stage.get_stage_script().contains("when t >= s0__at"), "the stage's script")
	print("%s: %d failures" % ["ok" if failures == 0 else "FAILED", failures])
	quit(failures)
