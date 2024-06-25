import unittest
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


class TestROIAlign(TestCase):

    def test_roi_align_amp(self):
        torch.manual_seed(1)
        torch.npu.manual_seed(1)
        x = torch.randn((1, 256, 200, 304)).npu().to(torch.float16)
        rois = torch.randn((648, 5)).npu()
        y = torch.ops.torchvision.roi_align(x.to(torch.float32), rois, 0.25, 7, 7, 2, False).to(torch.float16)
        with torch_npu.npu.amp.autocast():
            y_autocast = torch.ops.torchvision.roi_align(x, rois, 0.25, 7, 7, 2, False)
        self.assertEqual(y_autocast.dtype, torch.float16)
        self.assertRtolEqual(y, y_autocast)

    def test_faketensor_amp(self):
        with FakeTensorMode():
            with torch_npu.npu.amp.autocast():
                x = torch.randn((1, 256, 200, 304)).npu().to(torch.float16)
                rois = torch.randn((648, 5)).npu()
                y = torch.ops.torchvision.roi_align(x, rois, 0.25, 7, 7, 2, False)
        self.assertEqual(y.dtype, x.dtype)


if __name__ == '__main__':
    run_tests()
