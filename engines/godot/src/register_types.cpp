// The GDExtension's entry point: registers NeuralVFXScene and NeuralVFXEffect (nvfx_nodes.hpp).
#include "nvfx_nodes.hpp"

#include <gdextension_interface.h>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

namespace {

void initialize(ModuleInitializationLevel level) {
  if (level != MODULE_INITIALIZATION_LEVEL_SCENE) return;
  GDREGISTER_CLASS(NeuralVFXScene);
  GDREGISTER_CLASS(NeuralVFXEffect);
}

void uninitialize(ModuleInitializationLevel) {}

}  // namespace

extern "C" {

GDExtensionBool GDE_EXPORT neuralvfx_library_init(GDExtensionInterfaceGetProcAddress get_proc_address, GDExtensionClassLibraryPtr library,
                                                  GDExtensionInitialization* init) {
  GDExtensionBinding::InitObject init_obj(get_proc_address, library, init);
  init_obj.register_initializer(initialize);
  init_obj.register_terminator(uninitialize);
  init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
  return init_obj.init();
}

}  // extern "C"
