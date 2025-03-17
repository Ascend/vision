#ifndef __YUVP_RGB_H__
#define __YUVP_RGB_H__

#include <cinttypes>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <vector>

namespace vision {
namespace ops {

    enum KP_AVColorSpace {
        KP_AVCOL_SPC_RGB = 0,
        KP_AVCOL_SPC_BT709 = 1,
        KP_AVCOL_SPC_UNSPECIFIED = 2,
        KP_AVCOL_SPC_RESERVED = 3,
        KP_AVCOL_SPC_FCC = 4,
        KP_AVCOL_SPC_BT470BG = 5,
        KP_AVCOL_SPC_SMPTE170M = 6,
        KP_AVCOL_SPC_SMPTE240M = 7,
        KP_AVCOL_SPC_YCGCO = 8,
        KP_AVCOL_SPC_YCOCG = KP_AVCOL_SPC_YCGCO,
        KP_AVCOL_SPC_BT2020_NCL = 9,
        KP_AVCOL_SPC_BT2020_CL = 10,
        KP_AVCOL_SPC_SMPTE2085 = 11,
        KP_AVCOL_SPC_CHROMA_DERIVED_NCL = 12,
        KP_AVCOL_SPC_CHROMA_DERIVED_CL = 13,
        KP_AVCOL_SPC_ICTCP = 14,
        KP_AVCOL_SPC_IPT_C2 = 15,
        KP_AVCOL_SPC_YCGCO_RE = 16,
        KP_AVCOL_SPC_YCGCO_RO = 17,
        KP_AVCOL_SPC_NB,
    };

    enum KP_AVColorRange {
        KP_AVCOL_RANGE_UNSPECIFIED = 0,
        KP_AVCOL_RANGE_MPEG = 1,
        KP_AVCOL_RANGE_JPEG = 2,
        KP_AVCOL_RANGE_NB
    };

    extern "C" {
        void kp_yuv420p_to_rgb_limit(int ntasks, int ncols, int nrows, const uint8_t *ydata, int64_t *src_stride,
                                     const uint8_t *udata, const uint8_t *vdata, uint8_t *rgbdata);

        void kp_yuv420p_to_rgb_full(int ntasks, int ncols, int nrows, const uint8_t *ydata, int64_t *src_stride,
                                    const uint8_t *udata, const uint8_t *vdata, uint8_t *rgbdata);
    }
} // namespace ops
} // namespace vision


#endif  // __YUVP_RGB_H__