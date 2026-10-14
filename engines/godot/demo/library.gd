# The effect library in a window (docs/LIBRARY.md): a NeuralVFXStage with room for a score of the catalogue's effects.
# Left and right choose an effect, a click spawns it under the mouse (looping ones play until a right click stops the
# last), and the label says what it is. Run: godot --path engines/godot/demo res://library.tscn
extends Node2D

const Paths = preload("res://paths.gd")
const EFFECTS := ["campfire", "torch", "bonfire", "witch_fire", "steam_vent", "chimney_smoke", "grenade", "large_blast",
		"magic_burst", "portal", "healing_circle", "electric_arc", "lightning_strike", "muzzle_flash", "fireflies", "sparks",
		"flamethrower", "dust_impact", "ground_fog", "plasma_orb", "fire_whirl", "dust_devil"]

var stage := NeuralVFXStage.new()
var sprite := Sprite2D.new()
var label := Label.new()
var chosen := 0
var spawned: Array[int] = []


func _ready() -> void:
	var args := Paths.args()
	stage.library_dir = args.library
	stage.effects_dir = args.library_models
	stage.stage_size = Vector2i(1280, 720)
	stage.ground = 630
	var rooms := {}
	for e in EFFECTS:
		rooms[e] = [3, true]  # three of each at once; a fourth replaces the oldest
	stage.rooms = rooms
	stage.threads = 4
	add_child(stage)  # loads the stage (its learned effects warm up) and plays it
	sprite.centered = false
	sprite.texture = stage.texture
	add_child(sprite)
	add_child(label)
	if not stage.is_loaded():
		label.text = stage.get_error()


func _process(_delta: float) -> void:
	if stage.is_loaded():
		var info := stage.get_effect_info(EFFECTS[chosen])
		label.text = "%s (%s): %s\nleft/right: choose, click: spawn, right click: stop the last" % [info.name, info.category, info.description]


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_RIGHT:
			chosen = (chosen + 1) % EFFECTS.size()
		elif event.keycode == KEY_LEFT:
			chosen = (chosen + EFFECTS.size() - 1) % EFFECTS.size()
	elif event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_LEFT:
			var fx: int = stage.spawn(EFFECTS[chosen], get_local_mouse_position())
			if fx != 0:
				spawned.push_back(fx)
		elif event.button_index == MOUSE_BUTTON_RIGHT and not spawned.is_empty():
			stage.stop(spawned.pop_back())
