import sys

import numpy as np
import torch
import torch_npu
import torchvision.transforms as transforms
import torchvision_npu

import torch_npu.npu.utils as utils
from torchvision_npu.testing.test_deviation_case import TestCase
from torch_npu.testing.testcase import run_tests


class TestImageNormalize(TestCase):
    @staticmethod
    def cpu_op_exec(input1, mean, std):
        output = transforms.Normalize(mean=mean, std=std)(input1)
        return output

    @staticmethod
    def npu_op_exec(input1, mean, std):
        output = torch.ops.torchvision.npu_normalize(input1, mean, std)
        output = output.cpu().squeeze(0)
        return output

    def test_normalize(self):
        cpu_input = torch.from_numpy(np.random.uniform(0, 1, (3, 224, 224)).astype(np.float32))
        npu_input = cpu_input.unsqueeze(0).npu()
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]

        cpu_output = self.cpu_op_exec(cpu_input, mean, std)
        npu_output = self.npu_op_exec(npu_input, mean, std)

        self.assert_acceptable_deviation(npu_output, cpu_output, 1e-03)


if __name__ == "__main__":
    torch_npu.npu.set_device(0)
    if utils.get_soc_version() in range(220, 224):
        run_tests()
