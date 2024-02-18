import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision.transforms as transforms

import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestColorJitter(TestCase):
    def test_color_jitter(self):
        torch.npu.set_compile_mode(jit_compile=True)
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        npu_output = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)(npu_input)
        self.assertEqual(npu_output.device.type, 'npu')
        self.assertEqual(npu_output.dtype, torch.uint8)
        self.assertEqual(npu_output.shape, torch.Size([1, 3, 355, 432]))

    def test_adjust_hue(self):
        torch.ops.torchvision._dvpp_init()
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        factor = np.random.uniform(-0.5, 0.5)
        cpu_output = transforms.functional.adjust_hue(cpu_input, factor)
        npu_output = transforms.functional.adjust_hue(npu_input, factor).cpu().squeeze(0)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_adjust_contrast(self):
        torch.ops.torchvision._dvpp_init()
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        factor = np.random.uniform(0, 100)
        cpu_output = transforms.functional.adjust_contrast(cpu_input, factor)
        npu_output = transforms.functional.adjust_contrast(npu_input, factor).cpu().squeeze(0)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_adjust_brightness(self):
        torch.ops.torchvision._dvpp_init()
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        factor = np.random.uniform(0, 100)
        cpu_output = transforms.functional.adjust_brightness(cpu_input, factor)
        npu_output = transforms.functional.adjust_brightness(npu_input, factor).cpu().squeeze(0)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)
    
    def test_adjust_saturation(self):
        torch.ops.torchvision._dvpp_init()
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        factor = np.random.uniform(0, 100)
        cpu_output = transforms.functional.adjust_saturation(cpu_input, factor)
        npu_output = transforms.functional.adjust_saturation(npu_input, factor).cpu().squeeze(0)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == '__main__':
    run_tests()
