import unittest
import torch_npu
import torchvision
from torchvision import datasets, transforms
import torchvision_npu

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestAccelerate(unittest.TestCase):
    def test_accelerate_npu(self):
        path = "../Data/"
        torchvision.set_image_backend('npu')
        train_datasets = datasets.ImageFolder(path, transforms.RandomHorizontalFlip())
        self.assertTrue(train_datasets.accelerate_enable)
        self.assertNotEqual(train_datasets.device, "cpu")

    def test_accelerate_cpu(self):
        path = "../Data/"
        torchvision.set_image_backend('PIL')
        train_datasets = datasets.ImageFolder(path)
        self.assertFalse(train_datasets.accelerate_enable)
        self.assertEqual(train_datasets.device, "cpu")


if __name__ == '__main__':
    unittest.main()
