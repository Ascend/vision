import numpy as np

import torch
import torch_npu
import torchvision
import torchvision_npu

import torch_npu.npu.utils as utils
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuAdjustBrightness(TestCase):
    @staticmethod
    def cpu_op_exec(input1, factor):
        output = torchvision.transforms.functional.adjust_brightness(input1, factor)
        output = output.numpy()
        return output

    @staticmethod
    def npu_op_exec(input1, factor):
        output = torch.ops.torchvision.npu_adjust_brightness(input1, factor)
        output = output.cpu().squeeze(0)
        output = output.numpy()
        return output

    def test_npu_adjust_brightness(self):
        cpu_input = torch.from_numpy(np.random.uniform(0, 255, (3, 375, 500)).astype(np.uint8))
        npu_input = cpu_input.unsqueeze(0).npu()
        factor = 0.2

        cpu_output = self.cpu_op_exec(cpu_input, factor)
        npu_output = self.npu_op_exec(npu_input, factor)

        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    torch_npu.npu.set_device(0)
    if utils.get_soc_version() in range(220, 224):
        run_tests()
