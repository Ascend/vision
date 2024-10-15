import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision.transforms as transforms
import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase


class TestSolarize(TestCase):
    @staticmethod
    def cpu_op_exec(input1, threshold):
        output = transforms.functional.solarize(input1, threshold)
        return output

    @staticmethod
    def npu_op_exec(input1, threshold):
        output = transforms.functional.solarize(input1, threshold)
        output = output.cpu()
        return output

    def test_solarize_single(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        cpu_input = npu_input.cpu()

        threshold = 164

        cpu_output = self.cpu_op_exec(cpu_input, threshold)
        npu_output = self.npu_op_exec(npu_input, threshold)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_solarize_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)

        threshold = 0.6

        cpu_output = self.cpu_op_exec(cpu_input, threshold)
        npu_output = self.npu_op_exec(npu_input, threshold)
        self.assert_acceptable_deviation(npu_output, cpu_output, 0.2)

    def test_solarize_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)

        threshold = 164

        cpu_output = self.cpu_op_exec(cpu_input, threshold)
        npu_output = self.npu_op_exec(npu_input, threshold)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
