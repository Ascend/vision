import numpy as np
import torch
import torch_npu
import torchvision
from torch_npu.testing.testcase import TestCase, run_tests
import cv2
import av
import torchvision_npu

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


def CompareMaxDiff(img_1, img_2):
    img_1 = img_1.reshape(-1)
    img_2 = img_2.reshape(-1)
    max_diff = 0
    for index in range(img_1.size):
        pix_diff = 0
        if img_1[index] > img_2[index]:
            pix_diff = img_1[index] - img_2[index]
        else :
            pix_diff = img_2[index] - img_1[index]
        max_diff = max(max_diff, pix_diff)
    return max_diff


class TestDecodeVideo(TestCase):
    def test_decode_video_h264_one_frame(self):
        path = "../DataVideo/mountain/mountain.0001.h264"

        # pyav decode
        container = av.open(path)
        cpu_results = []
        for frame in container.decode(video=0):
            cpu_results.append(frame)
        container.close()

        # npu decode
        torch.npu.set_compile_mode(jit_compile=False)
        ret = torch.ops.torchvision._dvpp_sys_init()
        self.assertEqual(ret, 0)
        # HI_PT_H264 = 96, HI_PT_H265 = 265
        ptype = 96
        chn = torch.ops.torchvision._decode_video_create_chn(ptype)
        self.assertNotEqual(chn, -1)
        ret = torch.ops.torchvision._decode_video_start_get_frame(chn)
        self.assertEqual(ret, 0)

        with open(path, "rb") as f:
            f.seek(0)
            bytes_string = f.read()
            arr = np.frombuffer(bytes_string, dtype=np.uint8)
            input_tensor = torch.tensor(arr).npu(non_blocking=True)
            output_tensor = torch.zeros([1, 3, 1080, 1920], dtype=torch.uint8).npu(non_blocking=True) # NCHW
            outFormat = 69 # 12:rgb888; 13:bgr888; 69:rgb888planer; 70:bgr888planer
            ret = torch.ops.torchvision._decode_video_send_stream(chn, input_tensor, outFormat, True, output_tensor)
            self.assertEqual(ret, 0)

        npu_results = torch.ops.torchvision._decode_video_stop_get_frame(chn)
        self.assertEqual(len(npu_results), len(cpu_results))

        # check result
        paired = zip(npu_results, cpu_results)
        for pair in paired:
            npu_result = np.array(pair[0].cpu().permute(0, 2, 3, 1))
            cpu_result = pair[1].to_rgb().to_ndarray()
            max_diff = CompareMaxDiff(npu_result, cpu_result)
            self.assertLessEqual(max_diff, 3)

        ret = torch.ops.torchvision._decode_video_destroy_chnl(chn)
        self.assertEqual(ret, 0)
        ret = torch.ops.torchvision._dvpp_sys_exit()
        self.assertEqual(ret, 0)

    def test_decode_video_h264_multi_frame(self):
        path = "../DataVideo/billiards/billiards.0001.mp4"

        # pyav decode
        container = av.open(path)
        cpu_results = []
        for frame in container.decode(video=0):
            cpu_results.append(frame)
        container.close()

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
        ret = torch.ops.torchvision._decode_video_start_get_frame(chn)
        self.assertEqual(ret, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            input_tensor = torch.tensor(frame).npu(non_blocking=True)
            output_tensor = torch.zeros([1, 3, 480, 640], dtype=torch.uint8).npu(non_blocking=True) # NCHW
            outFormat = 69 # 12:rgb888; 13:bgr888; 69:rgb888planer; 70:bgr888planer
            ret = torch.ops.torchvision._decode_video_send_stream(chn, input_tensor, outFormat, True, output_tensor)
            self.assertEqual(ret, 0)

        npu_results = torch.ops.torchvision._decode_video_stop_get_frame(chn)
        self.assertEqual(len(npu_results), len(cpu_results)) # frame nums

        # check result
        paired = zip(npu_results, cpu_results)
        for pair in paired:
            npu_result = np.array(pair[0].cpu().permute(0, 2, 3, 1))
            cpu_result = pair[1].to_rgb().to_ndarray()
            max_diff = CompareMaxDiff(npu_result, cpu_result)
            self.assertLessEqual(max_diff, 3)

        ret = torch.ops.torchvision._decode_video_destroy_chnl(chn)
        self.assertEqual(ret, 0)
        ret = torch.ops.torchvision._dvpp_sys_exit()
        self.assertEqual(ret, 0)

    def test_decode_video_h265_one_frame(self):
        path = "../DataVideo/mountain/mountain.0001.h265"

        # pyav decode
        container = av.open(path)
        cpu_results = []
        for frame in container.decode(video=0):
            cpu_results.append(frame)
        container.close()

        # npu decode
        torch.npu.set_compile_mode(jit_compile=False)
        ret = torch.ops.torchvision._dvpp_sys_init()
        self.assertEqual(ret, 0)
        # HI_PT_H264 = 96, HI_PT_H265 = 265
        ptype = 265
        chn = torch.ops.torchvision._decode_video_create_chn(ptype)
        self.assertNotEqual(chn, -1)
        ret = torch.ops.torchvision._decode_video_start_get_frame(chn)
        self.assertEqual(ret, 0)

        with open(path, "rb") as f:
            f.seek(0)
            bytes_string = f.read()
            arr = np.frombuffer(bytes_string, dtype=np.uint8)
            input_tensor = torch.tensor(arr).npu(non_blocking=True)
            output_tensor = torch.zeros([1, 3, 1080, 1920], dtype=torch.uint8).npu(non_blocking=True) # NCHW
            outFormat = 69 # 12:rgb888; 13:bgr888; 69:rgb888planer; 70:bgr888planer
            ret = torch.ops.torchvision._decode_video_send_stream(chn, input_tensor, outFormat, True, output_tensor)
            self.assertEqual(ret, 0)

        npu_results = torch.ops.torchvision._decode_video_stop_get_frame(chn)
        self.assertEqual(len(npu_results), len(cpu_results))

        # check result
        paired = zip(npu_results, cpu_results)
        for pair in paired:
            npu_result = np.array(pair[0].cpu().permute(0, 2, 3, 1))
            cpu_result = pair[1].to_rgb().to_ndarray()
            max_diff = CompareMaxDiff(npu_result, cpu_result)
            self.assertLessEqual(max_diff, 3)

        ret = torch.ops.torchvision._decode_video_destroy_chnl(chn)
        self.assertEqual(ret, 0)
        ret = torch.ops.torchvision._dvpp_sys_exit()
        self.assertEqual(ret, 0)


if __name__ == '__main__':
    run_tests()
