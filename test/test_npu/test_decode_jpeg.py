import numpy as np

import torch
import torch_npu
import torchvision
import torchvision_npu

from torch_npu.testing.testcase import TestCase, run_tests

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestDecodeJpeg(TestCase):
    def test_decode_jpeg(self):
        path = "../Data/dog/dog.0001.jpg"
        cpu_output = torchvision.datasets.folder.pil_loader(path)
        cpu_output = torch.tensor(np.array(cpu_output)).permute(2, 0, 1).unsqueeze(0)
        npu_output = torchvision_npu.datasets.folder.npu_loader(path).cpu()
        self.assertEqual(npu_output, cpu_output)

    def test_decode_bmp(self):
        path = "../Data/bmp/pic.bmp"
        cpu_output = torchvision.datasets.folder.pil_loader(path)
        cpu_output = torch.tensor(np.array(cpu_output)).permute(2, 0, 1).unsqueeze(0)
        npu_output = torchvision_npu.datasets.folder.npu_loader(path).cpu()
        self.assertEqual(npu_output, cpu_output)

if __name__ == '__main__':
    run_tests()
