import sys

import numpy as np
import torch
import torch_npu
import torchvision.transforms as transforms
from torchvision_npu.extensions import _HAS_OPS

import torch_npu.npu.utils as utils
from torchvision_npu.testing.test_deviation_case import TestCase
from torch_npu.testing.testcase import run_tests


class TestCropAndResize(TestCase):
    @staticmethod
    def cpu_op_exec(input1, crop_factors, size, mode):
        top, left, height, width = \
            crop_factors[0], crop_factors[1], crop_factors[2], crop_factors[3]
        output = transforms.functional.resized_crop(input1,
            top, left, height, width, size, mode)
        return output

    @staticmethod
    def npu_op_exec(input1, crop_factors, size, mode):
        i, j, h, w = crop_factors[0], crop_factors[1], crop_factors[2], crop_factors[3]
        width, height = input1.shape[-1], input1.shape[-2]
        boxes = np.minimum([i / (height - 1), j / (width - 1), (i + h) / (height - 1),
            (j + w) / (width - 1)], 1).tolist()
        box_index = [0]
        crop_size = size

        output = torch.ops.torchvision.npu_crop_and_resize(input1,
            boxes=boxes, box_index=box_index, crop_size=crop_size, method=mode.value)

        output = output.cpu().squeeze(0)
        return output

    def test_npu_crop_and_resize(self):
        cpu_input = torch.from_numpy(np.random.uniform(0, 255, (3, 375, 500)).astype(np.uint8))
        npu_input = cpu_input.unsqueeze(0).npu()

        crop_factors = [10, 20, 150, 20]
        size = [224, 200]
        mode = transforms.InterpolationMode.BILINEAR

        cpu_output = self.cpu_op_exec(cpu_input, crop_factors, size, mode)
        npu_output = self.npu_op_exec(npu_input, crop_factors, size, mode)

        self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == "__main__":
    torch_npu.npu.set_device(0)
    if utils.get_soc_version() in range(220, 224):
        run_tests()
