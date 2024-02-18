import unittest
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
        output = output.cpu().squeeze(0)
        return output

    @unittest.expectedFailure
    def test_affine(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)

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
