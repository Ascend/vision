import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


class RotateAttr:
    def __init__(self, angle, interpolation, expand, center, fill): 
        self.angle = angle
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill


class TestRotate(TestCase):
    @staticmethod
    def cpu_op_exec(input1, attr):
        output = transforms.functional.rotate(input1,
            attr.angle, attr.interpolation, attr.expand, attr.center, attr.fill)
        return output

    @staticmethod
    def npu_op_exec(input1, attr):
        output = transforms.functional.rotate(input1,
            attr.angle, attr.interpolation, attr.expand, attr.center, attr.fill)
        output = output.cpu()
        return output

    @unittest.expectedFailure
    def test_rotate_single(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu()

        angle = -41.3
        center = [0, 0]
        fill = [0.0, 0.0, 0.0]

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            attr = RotateAttr(angle, interpolation, False, center, fill)
            cpu_output = self.cpu_op_exec(cpu_input, attr)
            npu_output = self.npu_op_exec(npu_input, attr)
            self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    @unittest.expectedFailure
    def test_rotate_expand_single(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu()

        angle = -41.3
        center = [int(npu_input.shape[3] / 2), int(npu_input.shape[2] / 2)]
        fill = [0.0, 0.0, 0.0]

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            attr = RotateAttr(angle, interpolation, True, center, fill)
            cpu_output = self.cpu_op_exec(cpu_input, attr)
            npu_output = self.npu_op_exec(npu_input, attr)
            self.assert_acceptable_deviation(npu_output, cpu_output, 2)
    
    @unittest.expectedFailure
    def test_rotate_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)

        angle = -41.3
        center = [0, 0]
        fill = [0.0, 0.0, 0.0]

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            attr = RotateAttr(angle, interpolation, False, center, fill)
            cpu_output = self.cpu_op_exec(cpu_input, attr)
            npu_output = self.npu_op_exec(npu_input, attr)
            self.assert_acceptable_deviation(npu_output, cpu_output, 2 / 255)

    @unittest.expectedFailure
    def test_rotate_expand_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)

        angle = -41.3
        center = [int(npu_input.shape[3] / 2), int(npu_input.shape[2] / 2)]
        fill = [0.0, 0.0, 0.0]

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            attr = RotateAttr(angle, interpolation, True, center, fill)
            cpu_output = self.cpu_op_exec(cpu_input, attr)
            npu_output = self.npu_op_exec(npu_input, attr)
            self.assert_acceptable_deviation(npu_output, cpu_output, 2 / 255)

    def test_rotate_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 224, 224), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)

        angle = 90
        center = [112, 112]
        fill = [0.0, 0.0, 0.0]

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            attr = RotateAttr(angle, interpolation, True, center, fill)
            cpu_output = self.cpu_op_exec(cpu_input, attr)
            npu_output = self.npu_op_exec(npu_input, attr)
            self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    @unittest.expectedFailure
    def test_rotate_expand_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)

        angle = -41.3
        center = [int(npu_input.shape[3] / 2), int(npu_input.shape[2] / 2)]
        fill = [0.0, 0.0, 0.0]

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            attr = RotateAttr(angle, interpolation, True, center, fill)
            cpu_output = self.cpu_op_exec(cpu_input, attr)
            npu_output = self.npu_op_exec(npu_input, attr)
            self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == '__main__':
    run_tests()
