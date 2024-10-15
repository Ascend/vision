import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision.transforms as transforms
import torchvision_npu


class TestFlip(TestCase):
    def test_horizontal_flip_single(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        cpu_output = transforms.RandomHorizontalFlip(p=1)(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.RandomHorizontalFlip(p=1)(npu_input).cpu().squeeze(0)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_vertical_flip_single(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        cpu_output = transforms.RandomVerticalFlip(p=1)(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.RandomVerticalFlip(p=1)(npu_input).cpu().squeeze(0)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_horizontal_flip_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)
        cpu_output = transforms.RandomHorizontalFlip(p=1)(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.RandomHorizontalFlip(p=1)(npu_input).cpu()
        self.assertRtolEqual(cpu_output, npu_output)

    def test_vertical_flip_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)
        cpu_output = transforms.RandomVerticalFlip(p=1)(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.RandomVerticalFlip(p=1)(npu_input).cpu()
        self.assertRtolEqual(cpu_output, npu_output)

    def test_horizontal_flip_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)
        cpu_output = transforms.RandomHorizontalFlip(p=1)(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.RandomHorizontalFlip(p=1)(npu_input).cpu()
        self.assertRtolEqual(cpu_output, npu_output)

    def test_vertical_flip_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)
        cpu_output = transforms.RandomVerticalFlip(p=1)(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.RandomVerticalFlip(p=1)(npu_input).cpu()
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
