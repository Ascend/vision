import unittest

import torch_npu
import torchvision_npu
from torchvision import datasets, transforms

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestAccelerate(unittest.TestCase):
    def test_accelerate(self):
        path = "../Data/"
        torchvision_npu.set_image_backend('npu')
        train_datasets = datasets.ImageFolder(path, transforms.RandomHorizontalFlip())
        self.assertTrue(train_datasets.accelerate_enable)
        self.assertNotEqual(train_datasets.device, "cpu")

    def test_accelerate(self):
        path = "../Data/"
        torchvision_npu.set_image_backend('PIL')
        train_datasets = datasets.ImageFolder(path)
        self.assertFalse(train_datasets.accelerate_enable)
        self.assertEqual(train_datasets.device, "cpu")


if __name__ == '__main__':
    unittest.main()
