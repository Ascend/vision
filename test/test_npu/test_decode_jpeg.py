import os
from pathlib import Path
import numpy as np
import torch
import torch_npu
import torchvision
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision_npu


torch_npu.npu.current_stream().set_data_preprocess_stream(True)
TEST_DIR = Path(__file__).resolve().parents[1]


class TestDecodeJpeg(TestCase):
    def test_decode_jpeg(self):
        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        cpu_output = torchvision.datasets.folder.pil_loader(path)
        cpu_output = torch.tensor(np.array(cpu_output)).permute(2, 0, 1).unsqueeze(0)

        torch.npu.set_compile_mode(jit_compile=True)
        npu_output = torchvision_npu.datasets._folder._npu_loader(path).cpu()
        self.assertEqual(npu_output, cpu_output)

        torch.npu.set_compile_mode(jit_compile=False)
        torch.ops.torchvision._dvpp_init()
        npu_output = torchvision_npu.datasets._folder._npu_loader(path).cpu()
        self.assertEqual(npu_output, cpu_output)

    def test_decode_bmp(self):
        path = os.path.join(TEST_DIR, "Data/bmp/pic.0001.bmp")
        cpu_output = torchvision.datasets.folder.pil_loader(path)
        cpu_output = torch.tensor(np.array(cpu_output)).permute(2, 0, 1).unsqueeze(0)
        npu_output = torchvision_npu.datasets._folder._npu_loader(path).cpu()
        self.assertEqual(npu_output, cpu_output)

if __name__ == '__main__':
    run_tests()
