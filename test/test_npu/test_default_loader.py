import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision
import torchvision_npu


class TestDefaultLoader(TestCase):

    def test_default_loader(self):
        path = "../Data/dog/dog.0001.jpg"
        torchvision.set_image_backend('PIL')
        img_output = torchvision.datasets.folder.default_loader(path)
        cpu_output = torchvision.transforms.functional.to_tensor(img_output)

        torchvision.set_image_backend('npu')
        npu_output = torchvision.datasets.folder.default_loader(path)
        npu_output = torchvision.transforms.functional.to_tensor(npu_output).squeeze(0)

        self.assertEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
