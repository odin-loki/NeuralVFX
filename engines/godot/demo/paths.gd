# Where the demo's files are: the arguments after "--" on Godot's command line (--script FILE --models DIR --out PNG
# --frames N --reference PNG), else the repository's fireball script and the study D effects in $NEURALVFX_DATA (default
# ~/nvfx-data).
extends RefCounted


static func args() -> Dictionary:
	var repo := ProjectSettings.globalize_path("res://").path_join("../../..").simplify_path()
	var data := OS.get_environment("NEURALVFX_DATA")
	if data.is_empty():
		data = OS.get_environment("HOME").path_join("nvfx-data")
	var out := {
		"script": repo.path_join("examples/scenes/fireball.nvfxs"),
		"models": data.path_join("experiments/models/d"),
		"out": "user://neuralvfx_frame.png",
		"frames": 45,
		"reference": "",
	}
	var a := OS.get_cmdline_user_args()
	var i := 0
	while i + 1 < a.size():
		var key: String = a[i].trim_prefix("--")
		if out.has(key):
			out[key] = int(a[i + 1]) if key == "frames" else a[i + 1]
		i += 2
	return out
