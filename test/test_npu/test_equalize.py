import os
from pathlib import Path
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision.transforms as transforms
import torchvision_npu


class TestEqualize(TestCase):
    @staticmethod
    def cpu_op_exec(input1):
        output = transforms.functional.equalize(input1)
        return output

    @staticmethod
    def npu_op_exec(input1):
        output = transforms.functional.equalize(input1)
        output = output.cpu()
        return output

    def test_equalize_multi_uint8(self):
        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)

        cpu_output = self.cpu_op_exec(cpu_input)
        npu_output = self.npu_op_exec(npu_input)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
