import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision.transforms as transforms
import torchvision_npu


class TestPosterize(TestCase):
    @staticmethod
    def cpu_op_exec(input1, bits):
        output = transforms.functional.posterize(input1, bits)
        return output

    @staticmethod
    def npu_op_exec(input1, bits):
        output = transforms.functional.posterize(input1, bits)
        output = output.cpu().squeeze(0)
        return output

    def test_posterize(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)

        for bits in range(9):
            cpu_output = self.cpu_op_exec(cpu_input, bits)
            npu_output = self.npu_op_exec(npu_input, bits)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
