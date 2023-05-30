import sys

import numpy as np
import torch
import torch_npu
import torchvision
import torchvision_npu

import torch_npu.npu.utils as utils
from torchvision_npu.testing.test_deviation_case import TestCase
from torch_npu.testing.testcase import run_tests


class TestNpuGaussianBlur(TestCase):
    @staticmethod
    def cpu_op_exec(input1, kernel_size, sigma):
        output = torchvision.transforms.functional.gaussian_blur(input1, kernel_size, sigma)
        return output

    @staticmethod
    def npu_op_exec(input1, kernel_size, sigma):
        output = torch.ops.torchvision.npu_gaussian_blur(input1, kernel_size, sigma, padding_mode="reflect")
        output = output.cpu().squeeze(0)
        return output

    def test_npu_gaussian_blur(self):
        cpu_input = torch.from_numpy(np.random.uniform(0, 255, (3, 375, 500)).astype(np.uint8))
        npu_input = cpu_input.unsqueeze(0).npu()
        kernel_size = (3, 5)
        sigma = (0.1, 2.0)

        cpu_output = self.cpu_op_exec(cpu_input, kernel_size, sigma)
        npu_output = self.npu_op_exec(npu_input, kernel_size, sigma)

        self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == "__main__":
    torch_npu.npu.set_device(0)
    if utils.get_soc_version() in range(220, 224):
        run_tests()
