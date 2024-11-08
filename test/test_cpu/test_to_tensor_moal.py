import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision
import torchvision_npu

 
class TestToTensorMoal(TestCase):
    def test_to_tensor_moal(self):
        batch_size = 256
        images = torch.randint(0, 255, (batch_size, 3, 1920, 1080), dtype=torch.uint8)

        # compute moal
        torchvision.set_image_backend('moal')
        to_tensor = torchvision.transforms.ToTensor()
        float_images_moal = to_tensor(images)

        # compute origin
        float_images_origin = images.float() / 255.0

        self.assertEqual(float_images_moal, float_images_origin)


if __name__ == '__main__':
    run_tests()
