import os
from pathlib import Path
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision.transforms as transforms
import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


TEST_DIR = Path(__file__).resolve().parents[1]


class TestCrop(TestCase):
    def test_center_crop_single(self):
        torch.ops.torchvision._dvpp_init()

        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        cpu_output = transforms.CenterCrop((100, 200))(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.CenterCrop((100, 200))(npu_input).cpu().squeeze(0)
        self.assertEqual(cpu_output, npu_output)

    def test_five_crop_single(self):
        torch.ops.torchvision._dvpp_init()

        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        cpu_output = transforms.FiveCrop((100))(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.FiveCrop((100))(npu_input)
        for i in range(5):
            npu_output_i = npu_output[i].cpu().squeeze(0)
            self.assertEqual(cpu_output[i], npu_output_i)

    def test_ten_crop_single(self):
        torch.ops.torchvision._dvpp_init()

        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        for is_vflip in [False, True]:
            cpu_output = transforms.TenCrop(50, is_vflip)(cpu_input)

            torch.npu.set_compile_mode(jit_compile=False)
            npu_output = transforms.TenCrop(50, is_vflip)(npu_input)
            for i in range(10):
                npu_output_i = npu_output[i].cpu().squeeze(0)
                self.assertEqual(cpu_output[i], npu_output_i)

    def test_center_crop_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 960, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)
        cpu_output = transforms.CenterCrop((100, 200))(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.CenterCrop((100, 200))(npu_input).cpu()
        self.assertEqual(cpu_output, npu_output)

    def test_five_crop_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 960, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)
        cpu_output = transforms.FiveCrop((100))(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.FiveCrop((100))(npu_input)
        for i in range(5):
            npu_output_i = npu_output[i].cpu()
            self.assertEqual(cpu_output[i], npu_output_i)

    def test_ten_crop_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 960, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)

        for is_vflip in [False, True]:
            cpu_output = transforms.TenCrop(50, is_vflip)(cpu_input)

            torch.npu.set_compile_mode(jit_compile=False)
            npu_output = transforms.TenCrop(50, is_vflip)(npu_input)
            for i in range(10):
                npu_output_i = npu_output[i].cpu()
                self.assertEqual(cpu_output[i], npu_output_i)

    def test_center_crop_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 960), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)
        cpu_output = transforms.CenterCrop((100, 200))(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.CenterCrop((100, 200))(npu_input).cpu()
        self.assertEqual(cpu_output, npu_output)

    def test_five_crop_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 960), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)
        cpu_output = transforms.FiveCrop((100))(cpu_input)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.FiveCrop((100))(npu_input)
        for i in range(5):
            npu_output_i = npu_output[i].cpu()
            self.assertEqual(cpu_output[i], npu_output_i)

    def test_ten_crop_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 960), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)

        for is_vflip in [False, True]:
            cpu_output = transforms.TenCrop(50, is_vflip)(cpu_input)

            torch.npu.set_compile_mode(jit_compile=False)
            npu_output = transforms.TenCrop(50, is_vflip)(npu_input)
            for i in range(10):
                npu_output_i = npu_output[i].cpu()
                self.assertEqual(cpu_output[i], npu_output_i)


if __name__ == '__main__':
    run_tests()
