import os
from pathlib import Path
import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision.transforms as transforms

import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


TEST_DIR = Path(__file__).resolve().parents[1]


class TestPad(TestCase):
    @staticmethod
    def cpu_op_exec(input1, padding, mode):
        output = transforms.Pad(padding, padding_mode=mode)(input1)
        output = output.numpy()
        return output

    @staticmethod
    def npu_op_exec(input1, padding, mode):
        output = transforms.Pad(padding, padding_mode=mode)(input1)
        output = output.cpu()
        output = output.numpy()
        return output

    def test_pad_single(self):
        torch.ops.torchvision._dvpp_init()

        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu()
        padding = (50, 100, 20, 70)
        for mode in ["constant", "edge"]:
            cpu_output = self.cpu_op_exec(cpu_input, padding, mode)

            torch.npu.set_compile_mode(jit_compile=False)
            npu_output = self.npu_op_exec(npu_input, padding, mode)
            self.assertEqual(cpu_output, npu_output)

    def test_pad_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)
        padding = (50, 100, 20, 70)
        for mode in ["constant", "edge"]:
            cpu_output = self.cpu_op_exec(cpu_input, padding, mode)

            torch.npu.set_compile_mode(jit_compile=False)
            npu_output = self.npu_op_exec(npu_input, padding, mode)
            self.assertEqual(cpu_output, npu_output)

    def test_pad_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)
        padding = (50, 100, 20, 70)
        for mode in ["constant", "edge"]:
            cpu_output = self.cpu_op_exec(cpu_input, padding, mode)

            torch.npu.set_compile_mode(jit_compile=False)
            npu_output = self.npu_op_exec(npu_input, padding, mode)
            self.assertEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
