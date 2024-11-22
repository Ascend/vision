import unittest
import torch
import torch.nn as nn

import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision
import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


class TestROIPool(TestCase):

    def test_roi_pool_amp(self):
        torch.manual_seed(1)
        torch.npu.manual_seed(1)
        feature_map = torch.rand(1, 512, 7, 7, dtype=torch.float16).npu()
        rois = torch.tensor([[0.0, 0.0, 0.0, 6.0, 6.0]]).npu()
        output_size = (6, 6)
        y = torchvision.ops.roi_pool(feature_map.to(torch.float32), rois, output_size).to(torch.float16)
        with torch_npu.npu.amp.autocast():
            y_autocast = torchvision.ops.roi_pool(feature_map, rois, output_size)
        self.assertEqual(y_autocast.dtype, torch.float16)
        self.assertEqual(y, y_autocast)

    def test_roi_pool_invalid_params(self):
        input_data = torch.randn((5, 5, 0, 0)).npu()
        rois = torch.randn((5, 5, 0, 0)).npu()
        output_size = [232, 813]
        with self.assertRaisesRegex(
            RuntimeError, "Expected input and rois to be non-empty tensors"
        ):
            torchvision.ops.roi_pool(input_data, rois, output_size)


if __name__ == '__main__':
    run_tests()
