import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision.transforms as transforms
import torchvision_npu


class TestInvert(TestCase):
    @staticmethod
    def cpu_op_exec(input1):
        output = transforms.functional.invert(input1)
        return output

    @staticmethod
    def npu_op_exec(input1):
        output = transforms.functional.invert(input1)
        output = output.cpu()
        return output

    def test_invert_single(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu()

        cpu_output = self.cpu_op_exec(cpu_input)
        npu_output = self.npu_op_exec(npu_input)
        self.assertEqual(cpu_output, npu_output)

    def test_invert_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 480, 360, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)

        cpu_output = self.cpu_op_exec(cpu_input)
        npu_output = self.npu_op_exec(npu_input)
        self.assertEqual(cpu_output, npu_output)

    def test_invert_multi_uint8(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.randint(0, 256, (4, 3, 480, 360), dtype=torch.uint8)
        npu_input = cpu_input.npu(non_blocking=True)

        cpu_output = self.cpu_op_exec(cpu_input)
        npu_output = self.npu_op_exec(npu_input)
        self.assertEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
