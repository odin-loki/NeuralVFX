#pragma once

namespace nfx::rt {

bool cpu_x86_has_avx2();
bool cpu_x86_has_avx512f();
bool cpu_x86_has_avx512_vnni();
bool cpu_x86_has_f16c();

}  // namespace nfx::rt
