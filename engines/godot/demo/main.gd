# The demo: the fireball scene script played by a NeuralVFXScene and drawn by a Sprite2D. The label shows the scene's
# heat under the mouse (a field the game reads); a click plays the scene again.
extends Node2D

const Paths = preload("res://paths.gd")

var scene := NeuralVFXScene.new()
var sprite := Sprite2D.new()
var label := Label.new()


func _ready() -> void:
	var args := Paths.args()
	scene.scene_script = args.script
	scene.effects_dir = args.models
	scene.threads = 4
	scene.looping = true
	add_child(scene)  # loads the scene
	sprite.centered = false
	sprite.texture = scene.texture
	add_child(sprite)
	add_child(label)
	if not scene.is_loaded():
		label.text = scene.get_error()


func _process(_delta: float) -> void:
	if scene.is_loaded():
		label.text = "t %.1f s   heat under the mouse %.2f" % [scene.get_time(), scene.get_heat(get_local_mouse_position())]


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.pressed:
		scene.restart()
