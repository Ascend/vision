#ifndef _COMMON_INCLUDE_H_
#define _COMMON_INCLUDE_H_
#include <iostream>
#include <vector>
#include <array>
#include <cstring>
#include <ctime>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <getopt.h>
#include <torch/torch.h>
#include <algorithm>
#include <omp.h>
 
namespace vision {
namespace ops {
#define OK (0)
#define FAIL (-1)
#define NS_TO_SEC (1000000000)
#define NEON_SIZE (4)
 
#ifndef LIKELY
#define LIKELY(x) (__builtin_expect(!!(x), 1) != 0)
#endif
 
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0) != 0)
#endif
}
}
 
#endif // _COMMON_INCLUDE_H_
