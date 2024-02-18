import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision.transforms as transforms
import torchvision_npu

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestCrop(TestCase):
    def test_center_crop(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        cpu_output = transforms.CenterCrop((100, 200))(cpu_input)

        torch.npu.set_compile_mode(jit_compile=True)
        npu_output = transforms.CenterCrop((100, 200))(npu_input).cpu().squeeze(0)
        self.assertEqual(cpu_output, npu_output)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.CenterCrop((100, 200))(npu_input).cpu().squeeze(0)
        self.assertEqual(cpu_output, npu_output)

    def test_five_crop(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        cpu_output = transforms.FiveCrop((100))(cpu_input)

        torch.npu.set_compile_mode(jit_compile=True)
        npu_output = transforms.FiveCrop((100))(npu_input)
        for i in range(5):
            npu_output_i = npu_output[i].cpu().squeeze(0)
            self.assertEqual(cpu_output[i], npu_output_i)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = transforms.FiveCrop((100))(npu_input)
        for i in range(5):
            npu_output_i = npu_output[i].cpu().squeeze(0)
            self.assertEqual(cpu_output[i], npu_output_i)

    def test_ten_crop(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        for is_vflip in [False, True]:
            cpu_output = transforms.TenCrop(50, is_vflip)(cpu_input)

            torch.npu.set_compile_mode(jit_compile=True)
            npu_output = transforms.TenCrop(50, is_vflip)(npu_input)
            for i in range(10):
                npu_output_i = npu_output[i].cpu().squeeze(0)
                self.assertEqual(cpu_output[i], npu_output_i)

            torch.npu.set_compile_mode(jit_compile=False)
            npu_output = transforms.TenCrop(50, is_vflip)(npu_input)
            for i in range(10):
                npu_output_i = npu_output[i].cpu().squeeze(0)
                self.assertEqual(cpu_output[i], npu_output_i)


if __name__ == '__main__':
    run_tests()
