import os
from pathlib import Path
import numpy as np
import torch
import torch_npu
import torchvision
from torchvision.io.image import ImageReadMode
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision_npu


TEST_DIR = Path(__file__).resolve().parents[1]
torch.ops.torchvision._dvpp_init()


class TestDecodeJpeg(TestCase):
    def test_npu_loader(self):
        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        cpu_output = torchvision.datasets.folder.pil_loader(path)
        cpu_output = torch.tensor(np.array(cpu_output)).permute(2, 0, 1).unsqueeze(0)

        torch.npu.set_compile_mode(jit_compile=True)
        npu_output = torchvision_npu.datasets._folder._npu_loader(path).cpu()
        self.assertEqual(npu_output, cpu_output)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = torchvision_npu.datasets._folder._npu_loader(path).cpu()
        self.assertEqual(npu_output, cpu_output)

    def test_decode_bmp(self):
        path = os.path.join(TEST_DIR, "Data/bmp/pic.0001.bmp")
        cpu_output = torchvision.datasets.folder.pil_loader(path)
        cpu_output = torch.tensor(np.array(cpu_output)).permute(2, 0, 1).unsqueeze(0)
        npu_output = torchvision_npu.datasets._folder._npu_loader(path).cpu()
        self.assertEqual(npu_output, cpu_output)

    def test_decode_jpeg(self):
        base_path = os.path.join(TEST_DIR, "Data")
        images_list = ['cat', 'dog', 'fish']
        for name in images_list:
            dir_path = os.path.join(base_path, name)
            for filename in os.listdir(dir_path):
                filepath = os.path.join(dir_path, filename)
                tensor_buf = torchvision.io.read_file(filepath)
                cpu_res = torchvision.io.decode_jpeg(tensor_buf)
                npu_res = torchvision.io.decode_jpeg(tensor_buf, device='npu').cpu()
                self.assertEqual(cpu_res, npu_res)

    def test_decode_jpeg_mode(self):
        base_path = os.path.join(TEST_DIR, "Data")
        images_list = ['cat', 'dog', 'fish']
        modes = [ImageReadMode.UNCHANGED, ImageReadMode.GRAY, ImageReadMode.RGB]
        for name in images_list:
            dir_path = os.path.join(base_path, name)
            for filename, mode in zip(os.listdir(dir_path), modes):
                filepath = os.path.join(dir_path, filename)
                tensor_buf = torchvision.io.read_file(filepath)
                
                cpu_res = torchvision.io.decode_jpeg(tensor_buf, mode=mode)
                npu_res = torchvision.io.decode_jpeg(tensor_buf, mode=mode, device='npu').cpu()
                self.assertEqual(cpu_res, npu_res)

if __name__ == '__main__':
    run_tests()
