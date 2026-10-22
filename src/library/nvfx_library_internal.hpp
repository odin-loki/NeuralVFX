// Inside the library only: what the effect pack's C API (src/pack/nvfx_pack.cpp) needs of the effect library's.
#pragma once

#include <neuralfx/nvfx_library.h>

#include <functional>
#include <string>

namespace nfx::library {

// Loads a learned effect's file, as the catalogue names it ("fire.nvfx"), from somewhere other than the models' folder.
using ModelSource = std::function<nvfx_status(const std::string& file, nvfx_effect** out)>;

// The stages of `library` load their learned effects through `source` (a pack's members) instead of from its models'
// folder. Called once, right after nvfx_library_open, before the library is shared.
void set_model_source(nvfx_library* library, ModelSource source);

}  // namespace nfx::library
