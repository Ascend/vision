#ifndef __KP_FAST_MEMCPY_H__
#define __KP_FAST_MEMCPY_H__

#include <cstddef>

extern "C" {
    void *kp_memcpy_fast(void *destination, const void *source, size_t size);
}

#endif  // __KP_FAST_MEMCPY_H__