import unittest
import torch
import torch.nn as nn
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


class TestROIAlign(TestCase):

    def test_roi_align_amp(self):
        linear = nn.Linear(304, 304).npu()
        x = torch.randn((1, 256, 200, 304)).npu()
        rois = torch.randn((648, 5)).npu()
        with torch_npu.npu.amp.autocast():
            x1 = linear(x)
            y = torch.ops.torchvision.roi_align(x1, rois, 0.25, 7, 7, 2, False)
        self.assertEqual(y.dtype, torch.float16)


if __name__ == '__main__':
    run_tests()
