import os
from pathlib import Path
import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


TEST_DIR = Path(__file__).resolve().parents[1]


class TestPerspective(TestCase):
    @staticmethod
    def cpu_op_exec(input1, startpoints, endpoints, interpolation, fill):
        output = transforms.functional.perspective(input1, startpoints, endpoints, interpolation, fill)
        return output

    @staticmethod
    def npu_op_exec(input1, startpoints, endpoints, interpolation, fill):
        output = transforms.functional.perspective(input1, startpoints, endpoints, interpolation, fill)
        output = output.cpu()
        return output

    def test_perspective_single(self):
        torch.ops.torchvision._dvpp_init()

        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu()

        h, w = npu_input.shape[-2], npu_input.shape[-1]

        startpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        endpoints = [[50, 0], [400, 100], [230, 340], [50, 250]]
        fill = [0.0, 0.0, 0.0]

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            cpu_output = self.cpu_op_exec(cpu_input, startpoints, endpoints, interpolation, fill)
            npu_output = self.npu_op_exec(npu_input, startpoints, endpoints, interpolation, fill)
            self.assertRtolEqual(npu_output, cpu_output)

    def test_perspective_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)

        h, w = npu_input.shape[-2], npu_input.shape[-1]

        startpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        endpoints = [[50, 0], [400, 100], [230, 340], [50, 250]]
        fill = [0.0, 0.0, 0.0]

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            cpu_output = self.cpu_op_exec(cpu_input, startpoints, endpoints, interpolation, fill)
            npu_output = self.npu_op_exec(npu_input, startpoints, endpoints, interpolation, fill)
            self.assertRtolEqual(npu_output, cpu_output)

    def test_perspective_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)

        h, w = npu_input.shape[-2], npu_input.shape[-1]

        startpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        endpoints = [[50, 0], [400, 100], [230, 340], [50, 250]]
        fill = [0.0, 0.0, 0.0]

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            cpu_output = self.cpu_op_exec(cpu_input, startpoints, endpoints, interpolation, fill)
            npu_output = self.npu_op_exec(npu_input, startpoints, endpoints, interpolation, fill)
            self.assertRtolEqual(npu_output, cpu_output, 2)


if __name__ == '__main__':
    run_tests()
