import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestRandomResizedCrop(TestCase):
    def test_resized_crop(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)

        top, left, height, width = 10, 20, 150, 200
        size = [224, 200]
        mode = InterpolationMode.BILINEAR

        cpu_output = transforms.functional.resized_crop(cpu_input,
            top, left, height, width, size, mode)

        torch.npu.set_compile_mode(jit_compile=True)
        npu_output = transforms.functional.resized_crop(npu_input,
            top, left, height, width, size, mode).cpu().squeeze(0)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.functional.resized_crop(npu_input,
            top, left, height, width, size, mode).cpu().squeeze(0)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == '__main__':
    run_tests()
