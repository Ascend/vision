import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision_npu
import torchvision.transforms as transforms

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestFlip(TestCase):
    def test_random_horizontal_flip(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        cpu_output = transforms.RandomHorizontalFlip(p=1)(cpu_input)
        npu_output = transforms.RandomHorizontalFlip(p=1)(npu_input).cpu().squeeze(0)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_random_vertical_flip(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        cpu_output = transforms.RandomVerticalFlip(p=1)(cpu_input)
        npu_output = transforms.RandomVerticalFlip(p=1)(npu_input).cpu().squeeze(0)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
