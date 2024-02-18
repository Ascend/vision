import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


class TestPerspective(TestCase):
    @staticmethod
    def cpu_op_exec(input1, startpoints, endpoints, interpolation, fill):
        output = transforms.functional.perspective(input1, startpoints, endpoints, interpolation, fill)
        return output

    @staticmethod
    def npu_op_exec(input1, startpoints, endpoints, interpolation, fill):
        output = transforms.functional.perspective(input1, startpoints, endpoints, interpolation, fill)
        output = output.cpu().squeeze(0)
        return output

    @unittest.expectedFailure
    def test_perspective(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)

        h, w = npu_input.shape[-2], npu_input.shape[-1]

        startpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        endpoints = [[50, 0], [400, 100], [230, 340], [50, 250]]
        fill = [0.0, 0.0, 0.0]

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            cpu_output = self.cpu_op_exec(cpu_input, startpoints, endpoints, interpolation, fill)
            npu_output = self.npu_op_exec(npu_input, startpoints, endpoints, interpolation, fill)
            self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == '__main__':
    run_tests()
