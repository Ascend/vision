#include "npu_decode_video_kernel.hpp"
#include <vector>
#include <map>
#include <sys/time.h>
#include <sys/prctl.h>
#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

namespace vision {
namespace ops {
constexpr uint32_t MAX_CHN_HEIGHT = 4096;
constexpr uint32_t MAX_CHN_WIDTH = 4096;
constexpr int32_t SEND_TIMEOUT = 30;
constexpr uint32_t WAIT_TIMEOUT = 5000000; // 5000000us
constexpr uint32_t REF_FRAME_NUM = 16;
constexpr uint32_t DISPLAY_FRAME_NUM = 16;
constexpr uint32_t FRAME_BUF_CNT = REF_FRAME_NUM + DISPLAY_FRAME_NUM + 1;

pthread_t g_vdec_get_thread[VDEC_MAX_CHNL_NUM] = {0};
uint32_t g_get_exit_state[VDEC_MAX_CHNL_NUM] = {0};
std::vector<std::vector<at::Tensor>> g_out_queue(VDEC_MAX_CHNL_NUM); // save success decoded frame
std::mutex outTensorMapMutex[VDEC_MAX_CHNL_NUM]; // map is not Thread-safe
std::map<hi_u64, at::Tensor> outTensorMap[VDEC_MAX_CHNL_NUM];

struct GetThreadPara {
    uint32_t chnId;
    uint32_t deviceId;
};

GetThreadPara g_getPara[VDEC_MAX_CHNL_NUM];

static inline bool ValidChnNum(uint32_t chn)
{
    return (chn < VDEC_MAX_CHNL_NUM);
}

static inline void get_current_time_us(uint64_t& timeUs)
{
    struct timeval curTime;
    gettimeofday(&curTime, NULL);
    timeUs = static_cast<uint64_t>(curTime.tv_sec) * 1000000 + curTime.tv_usec; // 1s = 1000000 us
}

template <class T>
static inline void LoadFunc(void* const handle, T& funPtr, const std::string& funName)
{
    funPtr = reinterpret_cast<T>(dlsym(handle, funName.c_str()));
    TORCH_CHECK(funPtr != nullptr, "vdec function not load, func name ", funName.c_str());
}

VideoDecode &VideoDecode::GetInstance()
{
    static VideoDecode instance;
    return instance;
}

VideoDecode::VideoDecode()
{
    LoadFunctions();
    for (uint32_t i = 0; i < VDEC_MAX_CHNL_NUM; ++i) {
        channelStatus_[i] = ChnStatus::DESTROYED;
    }
}

VideoDecode::~VideoDecode() {}

void VideoDecode::LoadFunctions()
{
    void * const handle  = dlopen("libacl_dvpp_mpi.so", RTLD_LAZY);
    TORCH_CHECK(handle != nullptr, "dlopen libacl_dvpp_mpi.so fail");

    LoadFunc(handle, sysInitFunPtr_, "hi_mpi_sys_init");
    LoadFunc(handle, sysExitFunPtr_, "hi_mpi_sys_exit");
    LoadFunc(handle, vdecGetPicBufferSizeFunPtr_, "hi_vdec_get_pic_buf_size");
    LoadFunc(handle, vdecGetTmvBufferSizeFunPtr_, "hi_vdec_get_tmv_buf_size");
    LoadFunc(handle, vdecCreateChnFunPtr_, "hi_mpi_vdec_create_chn");
    LoadFunc(handle, vdecDestroyChnFunPtr_, "hi_mpi_vdec_destroy_chn");
    LoadFunc(handle, vdecStartRecvStreamFunPtr_, "hi_mpi_vdec_start_recv_stream");
    LoadFunc(handle, vdecStopRecvStreamFunPtr_, "hi_mpi_vdec_stop_recv_stream");
    LoadFunc(handle, vdecQueryStatusFunPtr_, "hi_mpi_vdec_query_status");
    LoadFunc(handle, vdecResetChnFunPtr_, "hi_mpi_vdec_reset_chn");
    LoadFunc(handle, vdecSendStreamFunPtr_, "hi_mpi_vdec_send_stream");
    LoadFunc(handle, vdecGetFrameFunPtr_, "hi_mpi_vdec_get_frame");
    LoadFunc(handle, vdecReleaseFrameFunPtr_, "hi_mpi_vdec_release_frame");
}

int32_t VideoDecode::GetUnusedChn(uint32_t& chn)
{
    for (uint32_t i = 0; i < VDEC_MAX_CHNL_NUM; ++i) {
        const std::lock_guard<std::mutex> guard(channelMutex_[i]);
        if (channelStatus_[i] != ChnStatus::DESTROYED) {
            continue;
        } else {
            channelStatus_[i] = ChnStatus::CREATED;
            chn = i;
            return 0;
        }
    }
    return -1;
}

void VideoDecode::PutChn(uint32_t chn)
{
    const std::lock_guard<std::mutex> guard(channelMutex_[chn]);
    channelStatus_[chn] = ChnStatus::DESTROYED;
}

bool VideoDecode::ChannelCreated(uint32_t chn)
{
    const std::lock_guard<std::mutex> guard(channelMutex_[chn]);
    return (channelStatus_[chn] == ChnStatus::CREATED);
}

hi_s32 VideoDecode::sys_init(void)
{
    return sysInitFunPtr_();
}

hi_s32 VideoDecode::sys_exit(void)
{
    return sysExitFunPtr_();
}

hi_u32 VideoDecode::get_pic_buf_size(hi_payload_type type, hi_pic_buf_attr *buf_attr)
{
    return vdecGetPicBufferSizeFunPtr_(type, buf_attr);
}

hi_u32 VideoDecode::get_tmv_buf_size(hi_payload_type type, hi_u32 width, hi_u32 height)
{
    return vdecGetTmvBufferSizeFunPtr_(type, width, height);
}

hi_s32 VideoDecode::create_chn(hi_vdec_chn chn, const hi_vdec_chn_attr *attr)
{
    return vdecCreateChnFunPtr_(chn, attr);
}

hi_s32 VideoDecode::destroy_chn(hi_vdec_chn chn)
{
    return vdecDestroyChnFunPtr_(chn);
}

hi_s32 VideoDecode::start_recv_stream(hi_vdec_chn chn)
{
    return vdecStartRecvStreamFunPtr_(chn);
}

hi_s32 VideoDecode::stop_recv_stream(hi_vdec_chn chn)
{
    return vdecStopRecvStreamFunPtr_(chn);
}

hi_s32 VideoDecode::query_status(hi_vdec_chn chn, hi_vdec_chn_status *status)
{
    return vdecQueryStatusFunPtr_(chn, status);
}

hi_s32 VideoDecode::reset_chn(hi_vdec_chn chn)
{
    return vdecResetChnFunPtr_(chn);
}

hi_s32 VideoDecode::send_stream(hi_vdec_chn chn, const hi_vdec_stream *stream,
    hi_vdec_pic_info * vdec_pic_info, hi_s32 milli_sec)
{
    return vdecSendStreamFunPtr_(chn, stream, vdec_pic_info, milli_sec);
}

hi_s32 VideoDecode::get_frame(hi_vdec_chn chn, hi_video_frame_info *frame_info,
    hi_vdec_supplement_info *supplement, hi_vdec_stream *stream, hi_s32 milli_sec)
{
    return vdecGetFrameFunPtr_(chn, frame_info, supplement, stream, milli_sec);
}

hi_s32 VideoDecode::release_frame(hi_vdec_chn chn, const hi_video_frame_info *frame_info)
{
    return vdecReleaseFrameFunPtr_(chn, frame_info);
}

namespace {
static void vdec_reset_chn(uint32_t chn)
{
    int32_t ret = VideoDecode::GetInstance().stop_recv_stream(chn);
    TORCH_CHECK(ret == 0, "reset chn ", chn, ", hi_mpi_vdec_stop_recv_stream failed, ret = ", ret);

    ret = VideoDecode::GetInstance().reset_chn(chn);
    TORCH_CHECK(ret == 0, "reset chn ", chn, ", hi_mpi_vdec_reset_chn failed, ret = ", ret);

    ret = VideoDecode::GetInstance().start_recv_stream(chn);
    TORCH_CHECK(ret == 0, "reset chn ", chn, ", hi_mpi_vdec_start_recv_stream failed, ret = ", ret);
}

void* get_pic(void* args)
{
    prctl(PR_SET_NAME, "VdecGetPic", 0, 0, 0);
    GetThreadPara para = *(GetThreadPara*)args;
    uint32_t chanId = para.chnId;
    c10_npu::set_device(para.deviceId);

    int32_t ret = HI_SUCCESS;
    hi_video_frame_info frame{};
    hi_vdec_stream stream{};
    int32_t decResult = 0; // Decode result
    hi_u64 outputBuffer = 0;
    int32_t successCnt = 0;
    int32_t failCnt = 0;
    int32_t timeOut = 1000;

    g_get_exit_state[chanId] = 0;

    while (g_get_exit_state[chanId] == 0) {
        ret = VideoDecode::GetInstance().get_frame(chanId, &frame, nullptr, &stream, timeOut);
        if (ret == HI_SUCCESS) {
            // Flush decode end time
            outputBuffer = static_cast<hi_u64>(reinterpret_cast<uintptr_t>(frame.v_frame.virt_addr[0]));
            decResult = frame.v_frame.frame_flag;
            if (decResult == 0) { // 0: Decode success
                successCnt++;
                const std::lock_guard<std::mutex> guard(outTensorMapMutex[chanId]);
                auto iter = outTensorMap[chanId].find(outputBuffer);
                if (iter != outTensorMap[chanId].end()) {
                    // update size
                    auto outSize = iter->second.sizes();
                    std::vector<int64_t> outSizeNew(outSize.size());
                    for (uint32_t i = 0; i < outSize.size(); ++i) {
                        outSizeNew[i] = outSize[i];
                    }
                    int64_t format = at_npu::native::get_npu_format(iter->second); // 0:NCHW, 1:NHWC
                    if ((format == 0) || (format == 1)) {
                        if (format == 0) {
                            outSizeNew[2] = frame.v_frame.height;
                            outSizeNew[3] = frame.v_frame.width;
                        } else {
                            outSizeNew[1] = frame.v_frame.height;
                            outSizeNew[2] = frame.v_frame.width;
                        }
                        iter->second.resize_(outSizeNew);
                    }

                    g_out_queue[chanId].push_back(iter->second);
                    outTensorMap[chanId].erase(iter);
                }
            } else if (decResult == 1) { // 1: Decode fail
                failCnt++;
                TORCH_WARN("chn ", chanId, "GetFrame Success, decode failed, fail count ", failCnt);
            } else if (decResult == 2) {
                // 2:This result is returned for the second field of
                // the interlaced field stream, which is normal.
            } else if (decResult == 3) { // 3: Reference frame number set error
                failCnt++;
                TORCH_WARN("chn ", chanId, "GetFrame Success, refFrame num Error, fail count ", failCnt);
            } else if (decResult == 4) { // 4: Reference frame size set error
                failCnt++;
                TORCH_WARN("chn ", chanId, "GetFrame Success, refFrame Size Error, fail count ", failCnt);
            }

            // Release Frame
            ret = VideoDecode::GetInstance().release_frame(chanId, &frame);
            TORCH_CHECK(ret == 0, "chn ", chanId, ", hi_mpi_vdec_release_frame failed, ret = ", ret);
        } else {
            // 500us
            usleep(500);
        }
    }
    return (void*)HI_SUCCESS;
}

int64_t dvpp_sys_init()
{
    return static_cast<int64_t>(VideoDecode::GetInstance().sys_init());
}

int64_t dvpp_sys_exit()
{
    return static_cast<int64_t>(VideoDecode::GetInstance().sys_exit());
}

int64_t dvpp_vdec_create_chnl(int64_t pType)
{
    uint32_t chn = 0;
    int32_t ret = VideoDecode::GetInstance().GetUnusedChn(chn);
    TORCH_CHECK(ret == 0, "get unused chn failed");

    hi_vdec_chn_attr chnAttr{};
    chnAttr.type = static_cast<hi_payload_type>(pType);
    chnAttr.mode = HI_VDEC_SEND_MODE_FRAME; // Only support frame mode
    chnAttr.pic_width = MAX_CHN_WIDTH;
    chnAttr.pic_height = MAX_CHN_HEIGHT;
    chnAttr.stream_buf_size = MAX_CHN_WIDTH * MAX_CHN_HEIGHT * 3 / 2;
    chnAttr.frame_buf_cnt = FRAME_BUF_CNT;
    hi_pic_buf_attr buf_attr{MAX_CHN_WIDTH, MAX_CHN_HEIGHT, 0, HI_DATA_BIT_WIDTH_10, HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420,
        HI_COMPRESS_MODE_NONE};
    chnAttr.frame_buf_size = VideoDecode::GetInstance().get_pic_buf_size(chnAttr.type, &buf_attr);
    chnAttr.video_attr.ref_frame_num = REF_FRAME_NUM;
    chnAttr.video_attr.temporal_mvp_en = HI_TRUE;
    chnAttr.video_attr.tmv_buf_size = VideoDecode::GetInstance().get_tmv_buf_size(chnAttr.type, MAX_CHN_WIDTH,
        MAX_CHN_HEIGHT);

    ret = VideoDecode::GetInstance().create_chn(chn, &chnAttr);
    if (ret != HI_SUCCESS) {
        VideoDecode::GetInstance().PutChn(chn);
        AT_ERROR("hi_mpi_vdec_create_chn ", chn, ", failed, ret = ", ret);
        return -1;
    }

    ret = VideoDecode::GetInstance().start_recv_stream(chn);
    if (ret != HI_SUCCESS) {
        int32_t result = VideoDecode::GetInstance().destroy_chn(chn);
        VideoDecode::GetInstance().PutChn(chn);
        AT_ERROR("chn ", chn, ", hi_mpi_vdec_start_recv_stream failed, ret = ", ret);
        return -1;
    }

    return static_cast<int64_t>(chn);
}

int64_t dvpp_vdec_start_get_frame(int64_t chnId)
{
    TORCH_CHECK(ValidChnNum(chnId), "invalid chn ", chnId);

    int32_t deviceId = 0;
    aclError aclRet = c10_npu::GetDevice(&deviceId);
    TORCH_CHECK(aclRet == 0, "get device id failed, ret = ", aclRet);

    g_getPara[chnId].chnId = chnId;
    g_getPara[chnId].deviceId = deviceId;
    g_vdec_get_thread[chnId] = 0;
    int32_t ret = pthread_create(&g_vdec_get_thread[chnId], 0, get_pic, static_cast<void*>(&g_getPara[chnId]));
    if (ret != 0) {
        g_vdec_get_thread[chnId] = 0;
        AT_ERROR("Chn ", chnId, ", create get pic thread failed, ret = ", ret);
        return -1;
    }

    return 0;
}

int64_t dvpp_vdec_send_stream(int64_t chnId, const at::Tensor& self, int64_t outFormat, bool display, at::Tensor& out)
{
    TORCH_CHECK(ValidChnNum(chnId), "invalid chn ", chnId);
    hi_pixel_format outputFormat = static_cast<hi_pixel_format>(outFormat);
    TORCH_CHECK(((outputFormat == HI_PIXEL_FORMAT_RGB_888) || (outputFormat == HI_PIXEL_FORMAT_BGR_888) ||
                 (outputFormat == HI_PIXEL_FORMAT_RGB_888_PLANAR) || (outputFormat == HI_PIXEL_FORMAT_BGR_888_PLANAR)),
        "invalid outFormat ", outputFormat, ", should be ", HI_PIXEL_FORMAT_RGB_888, " or ", HI_PIXEL_FORMAT_BGR_888,
        " or ", HI_PIXEL_FORMAT_RGB_888_PLANAR, " or ", HI_PIXEL_FORMAT_BGR_888_PLANAR);

    int64_t format = at_npu::native::get_npu_format(out); // 0:NCHW, 1:NHWC
    TORCH_CHECK(((format == 0) || (format == 1)), "invalid npu format ", format);

    auto selfSize = self.sizes();
    int64_t selfNelements = c10::multiply_integers(selfSize);
    auto selfDtype = self.dtype();
    int64_t selfSizeBytes = selfNelements * selfDtype.itemsize();

    auto outSize = out.sizes();
    int64_t outNelements = c10::multiply_integers(outSize);
    auto outDtype = out.dtype();
    int64_t outSizeBytes = outNelements * outDtype.itemsize();

    hi_vdec_stream stream{};
    uint64_t currentSendTime = 0;
    get_current_time_us(currentSendTime);
    stream.pts = currentSendTime;
    stream.addr = static_cast<hi_u8*>(self.data_ptr());
    stream.len = selfSizeBytes;
    stream.end_of_frame = HI_TRUE;
    stream.end_of_stream = HI_FALSE;
    stream.need_display = display ? HI_TRUE : HI_FALSE;

    hi_vdec_pic_info outPicInfo{};
    outPicInfo.height = 0;
    outPicInfo.width = 0;
    outPicInfo.width_stride = 0;
    outPicInfo.height_stride = 0;
    outPicInfo.pixel_format = outputFormat;
    outPicInfo.vir_addr = 0;
    outPicInfo.buffer_size = 0;
    if (display) {
        outPicInfo.vir_addr = static_cast<hi_u64>(reinterpret_cast<uintptr_t>(out.data_ptr()));
        outPicInfo.buffer_size = outSizeBytes;
    }

    uint32_t sendOneFrameCnt = 0;
    int32_t ret = 0;
    do {
        sendOneFrameCnt++;
        // Send one frame data
        ret = VideoDecode::GetInstance().send_stream(chnId, &stream, &outPicInfo, SEND_TIMEOUT);
        if (sendOneFrameCnt > 30) { // if send stream timeout 30 times, end the decode process
            if (ret != 0) {
                vdec_reset_chn(chnId);
            }
            break;
        }
    } while (ret == HI_ERR_VDEC_BUF_FULL); // Try again
    TORCH_CHECK(ret == 0, "chn ", chnId, ", hi_mpi_vdec_send_stream failed, ret = ", ret);

    if (display) {
        const std::lock_guard<std::mutex> guard(outTensorMapMutex[chnId]);
        outTensorMap[chnId].insert(std::make_pair(static_cast<hi_u64>(reinterpret_cast<uintptr_t>(out.data_ptr())),
            out));
    }

    return 0;
}

std::vector<at::Tensor> dvpp_vdec_stop_get_frame(int64_t chnId)
{
    std::vector<at::Tensor> result;
    hi_vdec_chn_status status{};
    hi_vdec_chn_status pre_status{};

    hi_vdec_stream stream{};
    hi_vdec_pic_info outPicInfo{};
    // Send stream end flage
    stream.addr = NULL;
    stream.len = 0;
    stream.end_of_frame = HI_FALSE;
    stream.end_of_stream = HI_TRUE; // Stream end flage
    outPicInfo.vir_addr = 0;
    outPicInfo.buffer_size = 0;
    int32_t ret = VideoDecode::GetInstance().send_stream(chnId, &stream, &outPicInfo, -1);
    TORCH_CHECK(ret == 0, "chn ", chnId, ", hi_mpi_vdec_send_stream send end_of_stream failed, ret = ", ret);

    uint32_t waitTimes = 0;
    uint32_t sleepTime = 10000; // 10000us
    ret = VideoDecode::GetInstance().stop_recv_stream(chnId);
    TORCH_CHECK(ret == 0, "chn ", chnId, ", hi_mpi_vdec_stop_recv_stream failed, ret = ", ret);

    while (waitTimes < WAIT_TIMEOUT) {
        ret = VideoDecode::GetInstance().query_status(chnId, &status);
        TORCH_CHECK(ret == 0, "chn ", chnId, ", hi_mpi_vdec_query_status failed, ret = ", ret);
        if (((status.left_stream_bytes == 0) && (status.left_decoded_frames == 0))) {
            break;
        }
        if (status.left_decoded_frames == pre_status.left_decoded_frames) {
            waitTimes += sleepTime;
        } else {
            waitTimes = 0;
        }
        pre_status = status;
        // 10000us
        usleep(sleepTime);

        if (waitTimes >= WAIT_TIMEOUT) {
            vdec_reset_chn(chnId);
            break;
        }
    }

    g_get_exit_state[chnId] = 1; // notify get thread exit

    ret = pthread_join(g_vdec_get_thread[chnId], nullptr);
    TORCH_CHECK(ret == 0, "chn ", chnId, ", pthread_join get_pic thread failed, ret = ", ret);
    g_vdec_get_thread[chnId] = 0;

    for (const at::Tensor &tensor : g_out_queue[chnId]) {
        result.push_back(tensor);
    }

    g_out_queue[chnId].clear();
    outTensorMap[chnId].clear();
    return result;
}

int64_t dvpp_vdec_destroy_chnl(int64_t chnId)
{
    int32_t ret = VideoDecode::GetInstance().destroy_chn(chnId);
    VideoDecode::GetInstance().PutChn(chnId);
    TORCH_CHECK(ret == 0, "chn ", chnId, ", hi_mpi_vdec_destroy_chn failed, ret ", ret);
    return 0;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def("_dvpp_sys_init() -> int", &dvpp_sys_init);
    m.def("_dvpp_sys_exit() -> int", &dvpp_sys_exit);
    m.def("_decode_video_create_chn(int ptype) -> int", &dvpp_vdec_create_chnl);
    m.def("_decode_video_start_get_frame(int chnId) -> int", &dvpp_vdec_start_get_frame);
    m.def("_decode_video_send_stream(int chnId, Tensor self, int outFormat, bool display, Tensor out) -> int", &dvpp_vdec_send_stream);
    m.def("_decode_video_stop_get_frame(int chnId) -> Tensor[]", &dvpp_vdec_stop_get_frame);
    m.def("_decode_video_destroy_chnl(int chnId) -> int", &dvpp_vdec_destroy_chnl);
}
} // namespace ops
} // namespace vision
