import torch
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision.transforms as transforms

import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestGaussianBlur(TestCase):
    @staticmethod
    def cpu_op_exec(input1, kernel_size, sigma):
        output = transforms.functional.gaussian_blur(input1, kernel_size, sigma)
        return output

    @staticmethod
    def npu_op_exec(input1, kernel_size, sigma):
        output = transforms.functional.gaussian_blur(input1, kernel_size, sigma)
        output = output.cpu().squeeze(0)
        return output

    def test_gaussian_blur(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        kernel_size = (3, 5)
        sigma = (0.1, 2.0)

        cpu_output = self.cpu_op_exec(cpu_input, kernel_size, sigma)

        torch.npu.set_compile_mode(jit_compile=True)
        npu_output = self.npu_op_exec(npu_input, kernel_size, sigma)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = self.npu_op_exec(npu_input, kernel_size, sigma)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == '__main__':
    run_tests()
