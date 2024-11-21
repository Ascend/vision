import os
import unittest
from pathlib import Path
import torch
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


class AffineAttr:
    def __init__(self, angle, translate, scale, shear, interpolation, fill): 
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = interpolation
        self.fill = fill


class TestAffine(TestCase):
    @staticmethod
    def cpu_op_exec(input1, attr):
        output = transforms.functional.affine(input1,
            attr.angle, attr.translate, attr.scale, attr.shear, attr.interpolation, attr.fill)
        return output

    @staticmethod
    def npu_op_exec(input1, attr):
        output = transforms.functional.affine(input1,
            attr.angle, attr.translate, attr.scale, attr.shear, attr.interpolation, attr.fill)
        output = output.cpu()
        return output

    @unittest.expectedFailure
    def test_affine_single(self):
        torch.ops.torchvision._dvpp_init()

        path = os.path.join(Path(__file__).resolve().parents[1], "Data/dog/dog.0001.jpg")
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu()

        angle = 30.63
        translate = (-72, 42)
        scale = 0.6
        shear = (-7.4, 5.5)
        fill = [0.0, 0.0, 0.0]

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            attr = AffineAttr(angle, translate, scale, shear, interpolation, fill)
            cpu_output = self.cpu_op_exec(cpu_input, attr)
            npu_output = self.npu_op_exec(npu_input, attr)
            self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    @unittest.expectedFailure
    def test_affine_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)

        angle = 30.63
        translate = (-72, 42)
        scale = 0.6
        shear = (-7.4, 5.5)
        fill = [0.0, 0.0, 0.0]

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            attr = AffineAttr(angle, translate, scale, shear, interpolation, fill)
            cpu_output = self.cpu_op_exec(cpu_input, attr)
            npu_output = self.npu_op_exec(npu_input, attr)
            self.assert_acceptable_deviation(npu_output, cpu_output, 2 / 255)

    @unittest.expectedFailure
    def test_affine_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)

        angle = 30.63
        translate = (-72, 42)
        scale = 0.6
        shear = (-7.4, 5.5)
        fill = [0.0, 0.0, 0.0]

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]:
            attr = AffineAttr(angle, translate, scale, shear, interpolation, fill)
            cpu_output = self.cpu_op_exec(cpu_input, attr)
            npu_output = self.npu_op_exec(npu_input, attr)
            self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == '__main__':
    run_tests()
