import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


class TestRandomResizedCrop(TestCase):
    def test_resized_crop_single(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)

        top, left, height, width = 10, 20, 150, 200
        size = [224, 200]
        mode = InterpolationMode.BILINEAR

        cpu_output = transforms.functional.resized_crop(cpu_input,
            top, left, height, width, size, mode)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.functional.resized_crop(npu_input,
            top, left, height, width, size, mode).cpu().squeeze(0)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_resized_crop_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)

        top, left, height, width = 10, 20, 150, 200
        size = [224, 200]

        for mode in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]:
            cpu_output = transforms.functional.resized_crop(cpu_input,
                top, left, height, width, size, mode)

            torch.npu.set_compile_mode(jit_compile=False)
            npu_output = transforms.functional.resized_crop(npu_input,
                top, left, height, width, size, mode).cpu()
            self.assert_acceptable_deviation(npu_output, cpu_output, 2 / 255)

    def test_resized_crop_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)

        top, left, height, width = 10, 20, 150, 200
        size = [224, 200]

        for mode in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]:
            cpu_output = transforms.functional.resized_crop(cpu_input,
                top, left, height, width, size, mode)

            torch.npu.set_compile_mode(jit_compile=False)
            npu_output = transforms.functional.resized_crop(npu_input,
                top, left, height, width, size, mode).cpu()
            self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == '__main__':
    run_tests()
