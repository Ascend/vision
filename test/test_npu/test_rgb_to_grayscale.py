import os
from pathlib import Path
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision.transforms as transforms
import torchvision_npu


TEST_DIR = Path(__file__).resolve().parents[1]


class TestRgbToGrayscale(TestCase):
    @staticmethod
    def cpu_op_exec(input1, num_output_channels):
        output = transforms.functional.rgb_to_grayscale(input1, num_output_channels)
        return output

    @staticmethod
    def npu_op_exec(input1, num_output_channels):
        output = transforms.functional.rgb_to_grayscale(input1, num_output_channels)
        output = output.cpu()
        return output

    def test_rgb_to_grayscale_single(self):
        torch.ops.torchvision._dvpp_init()

        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu()

        for num_output_channels in [1, 3]:
            cpu_output = self.cpu_op_exec(cpu_input, num_output_channels)
            npu_output = self.npu_op_exec(npu_input, num_output_channels)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_rgb_to_grayscale_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)

        for num_output_channels in [1, 3]:
            cpu_output = self.cpu_op_exec(cpu_input, num_output_channels)
            npu_output = self.npu_op_exec(npu_input, num_output_channels)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_rgb_to_grayscale_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)

        for num_output_channels in [1, 3]:
            cpu_output = self.cpu_op_exec(cpu_input, num_output_channels)
            npu_output = self.npu_op_exec(npu_input, num_output_channels)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
