import unittest

import numpy as np
from PIL import Image
import cv2

import torch
import torch_npu
import torchvision_npu
import torchvision.transforms as transforms

from torchvision_npu.testing.test_deviation_case import TestCase
from torch_npu.testing.testcase import run_tests

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestResize(TestCase):
    def test_resize_vision(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        cpu_output = transforms.Resize((224, 224), Image.NEAREST)(cpu_input)
        npu_output = transforms.Resize((224, 224), Image.NEAREST)(npu_input).cpu().squeeze(0)
        self.assertEqual(cpu_output, npu_output)

    def test_resize_cv2(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = np.array(npu_input.cpu().squeeze(0).permute(1, 2, 0))
        size = (224, 200)
        interpolation_list = [[Image.NEAREST, cv2.INTER_NEAREST],
                              [Image.BILINEAR, cv2.INTER_LINEAR],
                              [Image.BICUBIC, cv2.INTER_CUBIC]]
        for i in range(3):
            cpu_output = cv2.resize(cpu_input, (size[1], size[0]), interpolation=interpolation_list[i][1])
            cpu_output = torch.from_numpy(cpu_output).permute(2, 0, 1)
            npu_output = transforms.Resize(size, interpolation=interpolation_list[i][0])(npu_input)
            npu_output = npu_output.cpu().squeeze(0)
            if i == 2:
                self.assert_acceptable_deviation(npu_output, cpu_output, 2)
            else:
                self.assertEqual(npu_output, cpu_output)


if __name__ == '__main__':
    run_tests()
