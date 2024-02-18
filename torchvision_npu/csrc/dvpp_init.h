#pragma once

#include <cstdint>
#include "macros.h"

namespace vision {

using DvppInitFunc = int (*)(const char *);
using DvppFinalizeFunc = int (*)();

VISION_API void dvpp_init();

namespace detail {
extern "C" VISION_INLINE_VARIABLE auto _register_ops = &dvpp_init;
#ifdef HINT_MSVC_LINKER_INCLUDE_SYMBOL
#pragma comment(linker, "/include:_register_ops")
#endif

} // namespace detail
} // namespace vision
