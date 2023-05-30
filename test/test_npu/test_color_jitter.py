import numpy as np

import torch
import torch_npu
import torchvision_npu
import torchvision.transforms as transforms

from torchvision_npu.testing.test_deviation_case import TestCase
from torch_npu.testing.testcase import run_tests

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestColorJitter(TestCase):
    def test_color_jitter(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        npu_output = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)(npu_input)
        self.assertEqual(npu_output.device.type, 'npu')
        self.assertEqual(npu_output.dtype, torch.uint8)
        self.assertEqual(npu_output.shape, torch.Size([1, 3, 355, 432]))

    def test_adjust_hue(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        factor = np.random.uniform(-0.5, 0.5)
        cpu_output = transforms.functional.adjust_hue(cpu_input, factor)
        npu_output = transforms.functional.adjust_hue(npu_input, factor).cpu().squeeze(0)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_adjust_contrast(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        factor = 1
        cpu_output = transforms.functional.adjust_contrast(cpu_input, factor)
        npu_output = transforms.functional.adjust_contrast(npu_input, factor).cpu().squeeze(0)
        self.assertEqual(cpu_output, npu_output)

    def test_adjust_brightness(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        factor = np.random.uniform(0, 100)
        cpu_output = transforms.functional.adjust_brightness(cpu_input, factor)
        npu_output = transforms.functional.adjust_brightness(npu_input, factor).cpu().squeeze(0)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)
    
    def test_adjust_saturation(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        factor = np.random.uniform(0, 1)
        cpu_output = transforms.functional.adjust_saturation(cpu_input, factor)
        npu_output = transforms.functional.adjust_saturation(npu_input, factor).cpu().squeeze(0)
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == '__main__':
    run_tests()
