import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision.transforms as transforms
import torchvision_npu

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestFlip(TestCase):
    def test_horizontal_flip(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        cpu_output = transforms.RandomHorizontalFlip(p=1)(cpu_input)

        torch.npu.set_compile_mode(jit_compile=True)
        npu_output = transforms.RandomHorizontalFlip(p=1)(npu_input).cpu().squeeze(0)
        self.assertRtolEqual(cpu_output, npu_output)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.RandomHorizontalFlip(p=1)(npu_input).cpu().squeeze(0)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_vertical_flip(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        cpu_output = transforms.RandomVerticalFlip(p=1)(cpu_input)

        torch.npu.set_compile_mode(jit_compile=True)
        npu_output = transforms.RandomVerticalFlip(p=1)(npu_input).cpu().squeeze(0)
        self.assertRtolEqual(cpu_output, npu_output)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.RandomVerticalFlip(p=1)(npu_input).cpu().squeeze(0)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
