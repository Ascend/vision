import time
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision
import torchvision_npu

 
class TestNormalizeMoal(TestCase):

    def test_normalize_moal(self):
        batch_size = 256
        images = torch.rand(batch_size, 3, 1920, 1080, dtype=torch.float)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        inplace = True
        # compute moal
        torchvision.set_image_backend('moal')
        start_time = time.perf_counter()
        normalize = torchvision.transforms.Normalize(mean, std, inplace)
        float_images_moal = normalize(images)
        end_time = time.perf_counter()
        normalize_time_moal = (end_time - start_time) * 1000

        # compute origin
        torchvision.set_image_backend('PIL')
        start_time = time.perf_counter()
        normalize = torchvision.transforms.Normalize(mean, std, inplace)
        float_images_origin = normalize(images)
        end_time = time.perf_counter()
        normalize_time_origin = (end_time - start_time) * 1000

        self.assertEqual(float_images_moal, float_images_origin)
        self.assertTrue(normalize_time_origin > normalize_time_moal)

        inplace = False
        # compute moal
        torchvision.set_image_backend('moal')
        start_time = time.perf_counter()
        normalize = torchvision.transforms.Normalize(mean, std, inplace)
        float_images_moal = normalize(images)
        end_time = time.perf_counter()
        normalize_time_moal = (end_time - start_time) * 1000

        # compute origin
        torchvision.set_image_backend('PIL')
        start_time = time.perf_counter()
        normalize = torchvision.transforms.Normalize(mean, std, inplace)
        float_images_origin = normalize(images)
        end_time = time.perf_counter()
        normalize_time_origin = (end_time - start_time) * 1000

        self.assertEqual(float_images_moal, float_images_origin)
        self.assertTrue(normalize_time_origin > normalize_time_moal)


if __name__ == '__main__':
    run_tests()
