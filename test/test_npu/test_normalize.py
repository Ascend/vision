import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision.transforms as transforms
import torchvision_npu


class TestNormalize(TestCase):
    @staticmethod
    def cpu_op_exec(input1, mean, std):
        output = transforms.Normalize(mean=mean, std=std)(input1)
        output = output.numpy()
        return output

    @staticmethod
    def npu_op_exec(input1, mean, std):
        output = transforms.Normalize(mean=mean, std=std)(input1)
        output = output.cpu()
        output = output.numpy()
        return output

    def test_normalize_single(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)
        npu_input = transforms.ToTensor()(npu_input)
        cpu_input = npu_input.cpu()
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]

        cpu_output = self.cpu_op_exec(cpu_input, mean, std)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = self.npu_op_exec(npu_input, mean, std)
        self.assertEqual(cpu_output, npu_output)

    def test_normalize_multi_float(self):
        torch.ops.torchvision._dvpp_init()

        cpu_input = torch.rand(4, 3, 224, 224, dtype=torch.float32)
        npu_input = cpu_input.npu(non_blocking=True)
        self.assertEqual(cpu_input, npu_input)

        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]

        cpu_output = self.cpu_op_exec(cpu_input, mean, std)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = self.npu_op_exec(npu_input, mean, std)
        self.assertEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
