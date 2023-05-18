import numpy as np
import cv2

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision_npu
import torchvision.transforms as transforms

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestGaussianBlur(TestCase):
    def result_error(self, npu_img, cpu_img):
        if npu_img.shape != cpu_img.shape:
            self.fail("shape error")
        if npu_img.dtype != cpu_img.dtype:
            self.fail("dtype error")
        result = np.abs(npu_img.to(torch.int16) - cpu_img.to(torch.int16))
        if result.max() > 2:
            self.fail("result error")

    def test_npu_gaussian_blur(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        kernel_size = (3, 5)
        sigma = (0.1, 2.0)

        cpu_output = transforms.functional.gaussian_blur(cpu_input, kernel_size, sigma)
        npu_output = transforms.functional.gaussian_blur(npu_input, kernel_size, sigma).cpu().squeeze(0)

        self.result_error(npu_output, cpu_output)


if __name__ == '__main__':
    run_tests()
