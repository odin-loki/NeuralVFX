# The headless test of the GDExtension (engines/godot/test.sh). A NeuralVFXScene plays the fireball script for N frames
# (45: 1.5 s, the fireball 0.3 s after the detonation) and its frame is saved as PNG. Checks: the frame is not empty
# (bright and varied) and, given the script runner's keyframe (--reference), the same to the bit; the heat is high where
# the fire is and zero far from it; the detonation rule fired once; a broken script is reported with its line and
# column; and a NeuralVFXEffect draws an effect. Exit code: the number of failed checks.
#
#   godot --headless --path engines/godot/demo --script res://test.gd -- --script FILE --models DIR --out PNG --frames 45
#         [--reference frame_045.png]
extends SceneTree

const Paths = preload("res://paths.gd")

var failures := 0


func check(ok: bool, what: String) -> void:
	print(("  ok    " if ok else "  FAIL  ") + what)
	if not ok:
		failures += 1


func _initialize() -> void:
	var args := Paths.args()
	print("NeuralVFX GDExtension test: %s, effects in %s" % [args.script, args.models])
	var scene := NeuralVFXScene.new()
	scene.scene_script = args.script
	scene.effects_dir = args.models
	scene.threads = 2
	scene.playing = false  # the frames are stepped here, not by the main loop
	var drawn := [0]
	scene.frame_drawn.connect(func(_frame: int) -> void: drawn[0] += 1)
	root.add_child(scene)
	scene.load_scene()  # (in a tree that runs, _ready() loads it)
	if not scene.is_loaded():
		printerr("cannot load the scene: ", scene.get_error())
		quit(1)
		return
	print("  %d x %d at %.0f fps, %.1f s; modules %s; rules %s" % [scene.get_scene_size().x, scene.get_scene_size().y, scene.get_fps(),
			scene.get_length(), scene.get_module_names(), scene.get_rule_names()])
	var frames: int = args.frames
	var t0 := Time.get_ticks_msec()
	for f in range(frames):
		scene.advance(1.0 / scene.get_fps())
	var ms: float = float(Time.get_ticks_msec() - t0) / maxi(1, frames)
	print("  %d frames, %.1f ms per frame (2 threads); frame %d at t %.2f s" % [frames, ms, scene.get_frame(), scene.get_time()])

	check(drawn[0] == frames + 1, "frame_drawn: frame 0 and every frame stepped (%d)" % drawn[0])
	check(scene.texture != null and scene.texture.get_width() == 1280, "the texture is the scene's picture")
	var img: Image = scene.get_image()
	check(img != null and img.get_width() == 1280 and img.get_height() == 720, "the frame is 1280 x 720")
	if img != null:
		check(img.save_png(args.out) == OK, "saved " + args.out)
		var sum := 0.0
		var bright := 0
		var samples := 0
		var colours := {}
		for y in range(0, img.get_height(), 8):
			for x in range(0, img.get_width(), 8):
				var c := img.get_pixel(x, y)
				var l := 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
				sum += l
				bright += 1 if l > 0.5 else 0
				samples += 1
				colours[c.to_rgba32() >> 12] = true
		print("  mean luminance %.3f, %d of %d samples bright, %d colours" % [sum / samples, bright, samples, colours.size()])
		check(sum / samples > 0.05, "the frame is not black")
		check(bright > 100, "the fireball is bright")
		check(colours.size() > 200, "the frame is not flat")
		if not str(args.reference).is_empty():
			var ref := Image.load_from_file(args.reference)
			if ref == null:
				check(false, "read the reference " + args.reference)
			else:
				ref.convert(Image.FORMAT_RGBA8)
				check(ref.get_data() == img.get_data(), "the same pixels as the script runner's frame (" + args.reference.get_file() + ")")

	var blast := scene.get_heat(Vector2(640, 500))
	var wreck := scene.get_field_max(NeuralVFXScene.FIELD_HEAT, Rect2(950, 480, 120, 120))
	var sky := scene.get_heat(Vector2(60, 60))
	var cell := scene.sample(Vector2(640, 500))
	print("  heat at the blast %.3f, over the wreck (max) %.3f, in the far sky %.3f; soot at the blast %.3f, velocity %s" % [blast, wreck, sky,
			cell.soot, cell.velocity])
	check(blast > 0.1, "heat where the fireball is")
	check(wreck > 0.1, "heat where the wreck burns")
	check(sky == 0.0, "no heat far from the fire")
	var grid := scene.get_field_grid(NeuralVFXScene.FIELD_HEAT, Vector2(16, 16), Vector2(32, 32), Vector2i(40, 22))
	var burning := 0
	for v in grid:
		burning += 1 if v > 0.1 else 0
	print("  %d of %d tiles of 32 px burning" % [burning, grid.size()])
	check(burning > 10 and burning < grid.size(), "a tile map of the heat: some tiles on fire")
	check(scene.get_rule_count("detonate") == 1, "the detonation rule fired once")
	var wreck_info := scene.get_module_info("wreck")
	print("  the wreck: ", wreck_info)
	check(wreck_info.active and wreck_info.position == Vector2(1010, 600), "the wreck's fire burns where the script put it")
	check(not scene.trigger("nonexistent"), "an unknown rule cannot be triggered")

	# a broken script: the error names the line and the column
	var bad := FileAccess.open("user://broken.nvfxs", FileAccess.WRITE)
	bad.store_string("scene size 320 x 180\nfield f = wind, velocity (1, 0), strenght 3\n")
	bad.close()
	var broken := NeuralVFXScene.new()
	broken.scene_script = "user://broken.nvfxs"
	root.add_child(broken)
	broken.load_scene()
	print("  broken script: ", broken.get_error())
	check(not broken.is_loaded() and broken.get_error_line() == 2 and broken.get_error_column() == 34, "a script error has its line and column")

	# one effect
	var fire := NeuralVFXEffect.new()
	fire.effect_path = args.models.path_join("fire.nvfx")
	fire.size = 128
	fire.controls = PackedFloat32Array([0.8, 0.5, 0.5])
	fire.seed = 7
	fire.playing = false
	root.add_child(fire)
	fire.load_effect()
	check(fire.is_loaded(), "NeuralVFXEffect loads fire.nvfx: " + str(fire.get_info()))
	if fire.is_loaded():
		fire.render_at(1.5)
		var e: Image = fire.get_image()
		var lit := 0
		for y in range(0, 128, 2):
			for x in range(0, 128, 2):
				lit += 1 if e.get_pixel(x, y).a > 0.1 else 0
		print("  the fire effect covers %d of 4096 samples" % lit)
		check(lit > 100, "the effect draws a fire")
	print("%s: %d failures" % ["ok" if failures == 0 else "FAILED", failures])
	quit(failures)
