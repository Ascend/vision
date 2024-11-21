import os
from pathlib import Path
import torch
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision.transforms as transforms

import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


TEST_DIR = Path(__file__).resolve().parents[1]


class TestGaussianBlur(TestCase):
    @staticmethod
    def cpu_op_exec(input1, kernel_size, sigma):
        output = transforms.functional.gaussian_blur(input1, kernel_size, sigma)
        return output

    @staticmethod
    def npu_op_exec(input1, kernel_size, sigma):
        output = transforms.functional.gaussian_blur(input1, kernel_size, sigma)
        output = output.cpu()
        return output

    def test_gaussian_blur_single(self):
        torch.ops.torchvision._dvpp_init()

        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu()
        kernel_size = (3, 5)
        sigma = (0.1, 2.0)

        cpu_output = self.cpu_op_exec(cpu_input, kernel_size, sigma)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = self.npu_op_exec(npu_input, kernel_size, sigma)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_gaussian_blur_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)
        kernel_size = (3, 5)
        sigma = (0.1, 2.0)

        cpu_output = self.cpu_op_exec(cpu_input, kernel_size, sigma)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = self.npu_op_exec(npu_input, kernel_size, sigma)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2 / 255)

    def test_gaussian_blur_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)
        kernel_size = (3, 5)
        sigma = (0.1, 2.0)

        cpu_output = self.cpu_op_exec(cpu_input, kernel_size, sigma)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = self.npu_op_exec(npu_input, kernel_size, sigma)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == '__main__':
    run_tests()
