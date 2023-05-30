import sys

import numpy as np
import torch
import torch_npu
import torchvision
import torchvision_npu

from torchvision_npu.testing.test_deviation_case import TestCase
from torch_npu.testing.testcase import run_tests


class TestNpuAdjustHue(TestCase):
    @staticmethod
    def cpu_op_exec(input1, factor):
        output = torchvision.transforms.functional.adjust_hue(input1, factor)
        return output

    @staticmethod
    def npu_op_exec(input1, factor):
        output = torch.ops.torchvision.npu_adjust_hue(input1.float(), factor)
        output = output.cpu().to(torch.uint8).squeeze(0)
        return output

    def test_npu_adjust_hue(self):
        cpu_input = torch.from_numpy(np.random.uniform(0, 255, (3, 375, 500)).astype(np.uint8))
        npu_input = cpu_input.unsqueeze(0).npu()
        factor = np.random.uniform(-0.5, 0.5)

        cpu_output = self.cpu_op_exec(cpu_input, factor)
        npu_output = self.npu_op_exec(npu_input, factor)

        self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == "__main__":
    run_tests()
