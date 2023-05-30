import numpy as np

import torch
import torch_npu
import torchvision_npu
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from torchvision_npu.testing.test_deviation_case import TestCase
from torch_npu.testing.testcase import run_tests

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestRandomResizedCrop(TestCase):
    def test_resized_crop(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)

        top, left, height, width = 10, 20, 150, 200
        size = [224, 200]
        mode = InterpolationMode.BILINEAR

        cpu_output = transforms.functional.resized_crop(cpu_input,
            top, left, height, width, size, mode)
        npu_output = transforms.functional.resized_crop(npu_input,
            top, left, height, width, size, mode).cpu().squeeze(0)
        
        self.assert_acceptable_deviation(npu_output, cpu_output, 2)

    def test_random_resized_crop_interpolation(self):
        path = "../Data/dog/dog.0001.jpg"
        img = torchvision_npu.datasets.folder.npu_loader(path)
        interpolation_list = [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]
        for i in interpolation_list:
            output = transforms.RandomResizedCrop(224, interpolation=i)(img)
            self.assertEqual(output.device.type, 'npu')
            self.assertEqual(output.dtype, torch.uint8)
            self.assertEqual(output.shape, torch.Size([1, 3, 224, 224]))


if __name__ == '__main__':
    run_tests()
