import numpy as np
import torch
import torch_npu
import torchvision
from torch_npu.testing.testcase import TestCase, run_tests
import cv2
import av
import torchvision_npu

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestDecodeVideo(TestCase):
    def _frames_compare(self, frames1, frames2):
        self.assertEqual(len(frames1), len(frames2))
        if len(frames1) == 0 or len(frames2) == 0:
            return
        self.assertEqual(frames1[0].shape, frames2[0].shape)

        max_diff_threshold = 3
        abs_diff = torch.where(frames1 > frames2, frames1 - frames2, frames2 - frames1)
        max_diff = abs_diff.max()
        self.assertLessEqual(max_diff, max_diff_threshold)

    def _get_frames_from_pyav(self, path) -> torch.Tensor:
        container = av.open(path)
        cpu_results = []
        for frame in container.decode(video=0):
            cpu_results.append(frame.to_rgb().to_ndarray())
        container.close()
        cpu_results = torch.as_tensor(np.stack(cpu_results)).permute(0, 3, 1, 2).npu(non_blocking=True)
        return cpu_results

    def test_decode_video_h264_one_frame(self):
        path = "../DataVideo/mountain/mountain.0001.h264"

        # pyav decode
        cpu_results = self._get_frames_from_pyav(path)

        # npu decode
        torch.npu.set_compile_mode(jit_compile=False)
        ret = torch.ops.torchvision._dvpp_sys_init()
        self.assertEqual(ret, 0)
        # HI_PT_H264 = 96, HI_PT_H265 = 265
        ptype = 96
        chn = torch.ops.torchvision._decode_video_create_chn(ptype)
        self.assertNotEqual(chn, -1)
        ret = torch.ops.torchvision._decode_video_start_get_frame(chn, 1)
        self.assertEqual(ret, 0)

        with open(path, "rb") as f:
            f.seek(0)
            bytes_string = f.read()
            arr = np.frombuffer(bytes_string, dtype=np.uint8)
            input_tensor = torch.tensor(arr).npu(non_blocking=True)
            output_tensor = torch.empty([1, 3, 1080, 1920], dtype=torch.uint8).npu(non_blocking=True) # NCHW
            outFormat = 69 # 12:rgb888; 13:bgr888; 69:rgb888planer; 70:bgr888planer
            ret = torch.ops.torchvision._decode_video_send_stream(chn, input_tensor, outFormat, True, output_tensor)
            self.assertEqual(ret, 0)

        torch.ops.torchvision._decode_video_stop_get_frame(chn, 1)
        self._frames_compare(output_tensor, cpu_results)
        ret = torch.ops.torchvision._decode_video_destroy_chnl(chn)
        self.assertEqual(ret, 0)
        ret = torch.ops.torchvision._dvpp_sys_exit()
        self.assertEqual(ret, 0)

    def test_decode_video_h264_multi_frame(self):
        path = "../DataVideo/billiards/billiards.0001.mp4"

        # pyav decode
        cpu_results = self._get_frames_from_pyav(path)

        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_FORMAT, -1)
        self.assertEqual(cap.isOpened(), True)

        # npu decode
        torch.npu.set_compile_mode(jit_compile=False)
        ret = torch.ops.torchvision._dvpp_sys_init()
        self.assertEqual(ret, 0)
        # HI_PT_H264 = 96, HI_PT_H265 = 265
        ptype = 96
        chn = torch.ops.torchvision._decode_video_create_chn(ptype)
        self.assertNotEqual(chn, -1)
        ret = torch.ops.torchvision._decode_video_start_get_frame(chn, cpu_results.size(0))
        self.assertEqual(ret, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            input_tensor = torch.tensor(frame).npu(non_blocking=True)
            output_tensor = torch.empty([3, 480, 640], dtype=torch.uint8).npu(non_blocking=True) # CHW
            outFormat = 69 # 12:rgb888; 13:bgr888; 69:rgb888planer; 70:bgr888planer
            ret = torch.ops.torchvision._decode_video_send_stream(chn, input_tensor, outFormat, True, output_tensor)
            self.assertEqual(ret, 0)

        npu_results = torch.ops.torchvision._decode_video_stop_get_frame(chn, cpu_results.size(0) + 1)
        self._frames_compare(npu_results, cpu_results)
        ret = torch.ops.torchvision._decode_video_destroy_chnl(chn)
        self.assertEqual(ret, 0)
        ret = torch.ops.torchvision._dvpp_sys_exit()
        self.assertEqual(ret, 0)

    def test_decode_video_h265_one_frame(self):
        path = "../DataVideo/mountain/mountain.0001.h265"

        # pyav decode
        cpu_results = self._get_frames_from_pyav(path)

        # npu decode
        torch.npu.set_compile_mode(jit_compile=False)
        ret = torch.ops.torchvision._dvpp_sys_init()
        self.assertEqual(ret, 0)
        # HI_PT_H264 = 96, HI_PT_H265 = 265
        ptype = 265
        chn = torch.ops.torchvision._decode_video_create_chn(ptype)
        self.assertNotEqual(chn, -1)
        ret = torch.ops.torchvision._decode_video_start_get_frame(chn, 1)
        self.assertEqual(ret, 0)

        with open(path, "rb") as f:
            f.seek(0)
            bytes_string = f.read()
            arr = np.frombuffer(bytes_string, dtype=np.uint8)
            input_tensor = torch.tensor(arr).npu(non_blocking=True)
            output_tensor = torch.empty([1, 3, 1080, 1920], dtype=torch.uint8).npu(non_blocking=True) # NCHW
            outFormat = 69 # 12:rgb888; 13:bgr888; 69:rgb888planer; 70:bgr888planer
            ret = torch.ops.torchvision._decode_video_send_stream(chn, input_tensor, outFormat, True, output_tensor)
            self.assertEqual(ret, 0)

        torch.ops.torchvision._decode_video_stop_get_frame(chn, 1)
        self._frames_compare(output_tensor, cpu_results)

        ret = torch.ops.torchvision._decode_video_destroy_chnl(chn)
        self.assertEqual(ret, 0)
        ret = torch.ops.torchvision._dvpp_sys_exit()
        self.assertEqual(ret, 0)


if __name__ == '__main__':
    run_tests()
