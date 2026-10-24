// Push/pop GCC function target for one ISA (rt_<isa>.cpp). Clang uses per-file -march in CMake (NEURALFX_WIN_CROSS).
#pragma once

#if defined(NFX_ARCH) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#define NFX_PRAGMA(x) _Pragma(#x)
#define NFX_TARGET(a) NFX_PRAGMA(GCC target(a))
NFX_TARGET(NFX_ARCH)
#define NFX_ISA_TARGET_PUSHED 1
#endif
