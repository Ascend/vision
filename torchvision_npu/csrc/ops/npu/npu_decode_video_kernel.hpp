/******************************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd
 * All rights reserved.
 *
 * Licensed under the BSD 3-Clause License  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef NPU_DECODE_VIDEO_KERNEL_HPP_
#define NPU_DECODE_VIDEO_KERNEL_HPP_

#include <cstdint>
#include <mutex>
#include <acl/dvpp/hi_dvpp.h>

namespace vision {
namespace ops {
constexpr uint32_t VDEC_MAX_CHNL_NUM = 32;

enum class ChnStatus {CREATED, DESTROYED};

class VideoDecode {
public:
    static VideoDecode &GetInstance();

    VideoDecode();
    ~VideoDecode();

    int32_t GetUnusedChn(uint32_t& chnl);
    void PutChn(uint32_t chnl);
    bool ChannelCreated(uint32_t chnl);

    hi_s32 sys_init(hi_void);
    hi_s32 sys_exit(hi_void);
    hi_u32 get_tmv_buf_size(hi_payload_type type, hi_u32 width, hi_u32 height);
    hi_u32 get_pic_buf_size(hi_payload_type type, hi_pic_buf_attr *buf_attr);
    hi_s32 create_chn(hi_vdec_chn chn, const hi_vdec_chn_attr *attr);
    hi_s32 destroy_chn(hi_vdec_chn chn);
    hi_s32 start_recv_stream(hi_vdec_chn chn);
    hi_s32 stop_recv_stream(hi_vdec_chn chn);
    hi_s32 query_status(hi_vdec_chn chn, hi_vdec_chn_status *status);
    hi_s32 reset_chn(hi_vdec_chn chn);
    hi_s32 send_stream(hi_vdec_chn chn, const hi_vdec_stream *stream, hi_vdec_pic_info *vdec_pic_info,
                       hi_s32 milli_sec);
    hi_s32 get_frame(hi_vdec_chn chn, hi_video_frame_info *frame_info, hi_vdec_supplement_info *supplement,
                     hi_vdec_stream *stream, hi_s32 milli_sec);
    hi_s32 release_frame(hi_vdec_chn chn, const hi_video_frame_info *frame_info);

private:
    std::mutex channelMutex_[VDEC_MAX_CHNL_NUM];
    ChnStatus channelStatus_[VDEC_MAX_CHNL_NUM];

    void LoadFunctions();
    hi_s32(*sysInitFunPtr_)(){nullptr};
    hi_s32(*sysExitFunPtr_)(){nullptr};
    hi_s32(*vdecCreateChnFunPtr_)(hi_vdec_chn, const hi_vdec_chn_attr*){nullptr};
    hi_s32(*vdecDestroyChnFunPtr_)(hi_vdec_chn){nullptr};
    hi_u32(*vdecGetPicBufferSizeFunPtr_)(hi_payload_type, hi_pic_buf_attr*){nullptr};
    hi_u32(*vdecGetTmvBufferSizeFunPtr_)(hi_payload_type, hi_u32, hi_u32){nullptr};
    hi_s32(*vdecStartRecvStreamFunPtr_)(hi_vdec_chn){nullptr};
    hi_s32(*vdecStopRecvStreamFunPtr_)(hi_vdec_chn){nullptr};
    hi_s32(*vdecQueryStatusFunPtr_)(hi_vdec_chn, hi_vdec_chn_status*){nullptr};
    hi_s32(*vdecResetChnFunPtr_)(hi_vdec_chn){nullptr};
    hi_s32(*vdecSendStreamFunPtr_)(hi_vdec_chn, const hi_vdec_stream*, hi_vdec_pic_info*, hi_s32){nullptr};
    hi_s32(*vdecGetFrameFunPtr_)(hi_vdec_chn, hi_video_frame_info*, hi_vdec_supplement_info*, hi_vdec_stream*,
                                 hi_s32){nullptr};
    hi_s32(*vdecReleaseFrameFunPtr_)(hi_vdec_chn, const hi_video_frame_info*){nullptr};
};
} // namespace ops
} // namespace vision

#endif // NPU_DECODE_VIDEO_KERNEL_HPP_
