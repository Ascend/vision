import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision.transforms as transforms
import torchvision_npu


class TestSolarize(TestCase):
    @staticmethod
    def cpu_op_exec(input1, threshold):
        output = transforms.functional.solarize(input1, threshold)
        return output

    @staticmethod
    def npu_op_exec(input1, threshold):
        output = transforms.functional.solarize(input1, threshold)
        output = output.cpu().squeeze(0)
        return output

    def test_solarize(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)

        threshold = 164

        cpu_output = self.cpu_op_exec(cpu_input, threshold)
        npu_output = self.npu_op_exec(npu_input, threshold)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
