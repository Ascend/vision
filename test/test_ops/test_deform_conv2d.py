import unittest
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

import torch_npu
from torch_npu.testing.testcase import run_tests
from torchvision.ops import deform_conv2d
import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


class TestDeformConv2d(TestCase):

    def test_deform_conv2d(self):
        torch.manual_seed(1)
        torch.npu.manual_seed(1)
        input_tensor = torch.randn(1, 3, 8, 8)
        offset = torch.randn(1, 18, 8, 8)
        weight = torch.randn(5, 3, 3, 3)  
        bias = torch.randn(5)
        cpu_output = deform_conv2d(input_tensor, offset, weight, bias=bias, stride=1, padding=1)
        npu_output = deform_conv2d(input_tensor.npu(), offset.npu(), weight.npu(), bias=bias.npu(), stride=1, padding=1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_deform_conv2d_kwargs(self):
        torch.manual_seed(1)
        torch.npu.manual_seed(1)
        input_tensor = torch.randn(1, 3, 8, 8)
        offset = torch.randn(1, 18, 8, 8)
        weight = torch.randn(5, 3, 3, 3)  
        bias = torch.randn(5)
        cpu_output = deform_conv2d(input=input_tensor, offset=offset, weight=weight, bias=bias, stride=1, padding=1)
        npu_output = deform_conv2d(input=input_tensor.npu(), offset=offset.npu(), weight=weight.npu(), bias=bias.npu(), stride=1, padding=1)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
