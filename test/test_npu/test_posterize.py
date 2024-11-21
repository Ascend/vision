import os
from pathlib import Path
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision.transforms as transforms
import torchvision_npu


TEST_DIR = Path(__file__).resolve().parents[1]


class TestPosterize(TestCase):
    @staticmethod
    def cpu_op_exec(input1, bits):
        output = transforms.functional.posterize(input1, bits)
        return output

    @staticmethod
    def npu_op_exec(input1, bits):
        output = transforms.functional.posterize(input1, bits)
        output = output.cpu()
        return output

    def test_posterize_single(self):
        torch.ops.torchvision._dvpp_init()

        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu()

        for bits in range(9):
            cpu_output = self.cpu_op_exec(cpu_input, bits)
            npu_output = self.npu_op_exec(npu_input, bits)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_posterize_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)

        for bits in range(9):
            cpu_output = self.cpu_op_exec(cpu_input, bits)
            npu_output = self.npu_op_exec(npu_input, bits)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
