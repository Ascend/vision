import os
from pathlib import Path
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision.transforms as transforms

import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


TEST_DIR = Path(__file__).resolve().parents[1]


class TestColorJitter(TestCase):
    def test_color_jitter(self):
        torch.npu.set_compile_mode(jit_compile=True)
        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        npu_output = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)(npu_input)
        self.assertEqual(npu_output.device.type, 'npu')
        self.assertEqual(npu_output.dtype, torch.uint8)
        self.assertEqual(npu_output.shape, torch.Size([1, 3, 355, 432]))

    def test_adjust_hue_single(self):
        torch.ops.torchvision._dvpp_init()
        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        factor = np.random.uniform(-0.5, 0.5)
        cpu_output = transforms.functional.adjust_hue(cpu_input, factor)
        npu_output = transforms.functional.adjust_hue(npu_input, factor).cpu().squeeze(0)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_adjust_contrast_single(self):
        torch.ops.torchvision._dvpp_init()
        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        factor = np.random.uniform(0, 100)
        cpu_output = transforms.functional.adjust_contrast(cpu_input, factor)
        npu_output = transforms.functional.adjust_contrast(npu_input, factor).cpu().squeeze(0)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_adjust_brightness_single(self):
        torch.ops.torchvision._dvpp_init()
        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        factor = np.random.uniform(0, 100)
        cpu_output = transforms.functional.adjust_brightness(cpu_input, factor)
        npu_output = transforms.functional.adjust_brightness(npu_input, factor).cpu().squeeze(0)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)
    
    def test_adjust_saturation_single(self):
        torch.ops.torchvision._dvpp_init()
        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        factor = np.random.uniform(0, 100)
        cpu_output = transforms.functional.adjust_saturation(cpu_input, factor)
        npu_output = transforms.functional.adjust_saturation(npu_input, factor).cpu().squeeze(0)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_adjust_hue_multi_float(self):
        torch.ops.torchvision._dvpp_init()
        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)
        factor = np.random.uniform(-0.5, 0.5)
        cpu_output = transforms.functional.adjust_hue(cpu_input, factor)
        npu_output = transforms.functional.adjust_hue(npu_input, factor).cpu()
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_adjust_contrast_multi_float(self):
        torch.ops.torchvision._dvpp_init()
        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)
        factor = np.random.uniform(0, 100)
        cpu_output = transforms.functional.adjust_contrast(cpu_input, factor)
        npu_output = transforms.functional.adjust_contrast(npu_input, factor).cpu()
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_adjust_brightness_multi_float(self):
        torch.ops.torchvision._dvpp_init()
        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)
        factor = np.array([1.5])
        cpu_output = transforms.functional.adjust_brightness(cpu_input, factor)
        npu_output = transforms.functional.adjust_brightness(npu_input, factor).cpu()
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)
    
    def test_adjust_saturation_multi_float(self):
        torch.ops.torchvision._dvpp_init()
        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)
        factor = np.random.uniform(0, 100)
        cpu_output = transforms.functional.adjust_saturation(cpu_input, factor)
        npu_output = transforms.functional.adjust_saturation(npu_input, factor).cpu()
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_adjust_hue_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()
        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)
        factor = np.random.uniform(-0.5, 0.5)
        cpu_output = transforms.functional.adjust_hue(cpu_input, factor)
        npu_output = transforms.functional.adjust_hue(npu_input, factor).cpu()
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_adjust_contrast_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()
        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)
        factor = np.random.uniform(0, 100)
        cpu_output = transforms.functional.adjust_contrast(cpu_input, factor)
        npu_output = transforms.functional.adjust_contrast(npu_input, factor).cpu()
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_adjust_brightness_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()
        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)
        factor = np.random.uniform(0, 100)
        cpu_output = transforms.functional.adjust_brightness(cpu_input, factor)
        npu_output = transforms.functional.adjust_brightness(npu_input, factor).cpu()
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)
    
    def test_adjust_saturation_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()
        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)
        factor = np.random.uniform(0, 100)
        cpu_output = transforms.functional.adjust_saturation(cpu_input, factor)
        npu_output = transforms.functional.adjust_saturation(npu_input, factor).cpu()
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == '__main__':
    run_tests()
