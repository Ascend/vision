import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


class TestResize(TestCase):
    def test_resize_vision_single(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]:
            cpu_output = transforms.Resize((224, 224), interpolation)(cpu_input)

            torch.npu.set_compile_mode(jit_compile=False)
            npu_output = transforms.Resize((224, 224), interpolation)(npu_input).cpu().squeeze(0)
            self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    @unittest.expectedFailure
    def test_resize_vision_multi_float_nearest(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)

        interpolation = InterpolationMode.NEAREST
        cpu_output = transforms.Resize((224, 224), interpolation)(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.Resize((224, 224), interpolation)(npu_input).cpu()
        self.assert_acceptable_deviation(npu_output, cpu_output, 2 / 255)   
    
    def test_resize_vision_multi_float_bilinear(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)

        interpolation = InterpolationMode.BILINEAR
        cpu_output = transforms.Resize((224, 224), interpolation)(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.Resize((224, 224), interpolation)(npu_input).cpu()
        self.assert_acceptable_deviation(npu_output, cpu_output, 2 / 255)

    def test_resize_vision_multi_float_bicubic(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)

        interpolation = InterpolationMode.BICUBIC
        cpu_output = transforms.Resize((224, 224), interpolation)(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.Resize((224, 224), interpolation)(npu_input).cpu()
        self.assert_acceptable_deviation(npu_output, cpu_output, 2 / 255)

    @unittest.expectedFailure
    def test_resize_vision_multi_uint8_nearest(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)

        interpolation = InterpolationMode.NEAREST
        cpu_output = transforms.Resize((224, 224), interpolation)(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.Resize((224, 224), interpolation)(npu_input).cpu()
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)   
    
    def test_resize_vision_multi_uint8_bilinear(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)

        interpolation = InterpolationMode.BILINEAR
        cpu_output = transforms.Resize((224, 224), interpolation)(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.Resize((224, 224), interpolation)(npu_input).cpu()
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_resize_vision_multi_uint8_bicubic(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)

        interpolation = InterpolationMode.BICUBIC
        cpu_output = transforms.Resize((224, 224), interpolation)(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.Resize((224, 224), interpolation)(npu_input).cpu()
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == '__main__':
    run_tests()
