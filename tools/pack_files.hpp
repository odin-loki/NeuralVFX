// nvfx_pack's modes for effect packs (src/pack/pack.hpp, include/neuralfx/nvfx_pack.h): pack, list, verify, extract,
// and the measurement that chose the default coder (docs/INTEGRATION.md §5).
#pragma once

#include "args.hpp"

#include <string_view>

namespace nfx::tools {

// True if the first word names a pack mode (pack, list, verify, extract, measure).
bool is_pack_mode(std::string_view word);
// Runs the mode; returns the exit status (77: the effects' files are missing and --require-models was given).
int pack_mode(const Args& a);

}  // namespace nfx::tools
