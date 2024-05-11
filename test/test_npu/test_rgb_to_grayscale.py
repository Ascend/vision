import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision.transforms as transforms
import torchvision_npu


class TestRgbToGrayscale(TestCase):
    @staticmethod
    def cpu_op_exec(input1, num_output_channels):
        output = transforms.functional.rgb_to_grayscale(input1, num_output_channels)
        return output

    @staticmethod
    def npu_op_exec(input1, num_output_channels):
        output = transforms.functional.rgb_to_grayscale(input1, num_output_channels)
        output = output.cpu().squeeze(0)
        return output

    def test_rgb_to_grayscale(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)

        for num_output_channels in [1, 3]:
            cpu_output = self.cpu_op_exec(cpu_input, num_output_channels)
            npu_output = self.npu_op_exec(npu_input, num_output_channels)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
