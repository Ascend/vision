import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision.transforms as transforms
import torchvision_npu

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestNormalize(TestCase):
    @staticmethod
    def cpu_op_exec(input1, mean, std):
        output = transforms.Normalize(mean=mean, std=std)(input1)
        output = output.numpy()
        return output

    @staticmethod
    def npu_op_exec(input1, mean, std):
        output = transforms.Normalize(mean=mean, std=std)(input1)
        output = output.cpu().squeeze(0)
        output = output.numpy()
        return output

    def test_normalize(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        npu_input = transforms.ToTensor()(npu_input)
        cpu_input = npu_input.cpu().squeeze(0)
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]

        cpu_output = self.cpu_op_exec(cpu_input, mean, std)

        torch.npu.set_compile_mode(jit_compile=True)
        npu_output = self.npu_op_exec(npu_input, mean, std)
        self.assertEqual(cpu_output, npu_output)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = self.npu_op_exec(npu_input, mean, std)
        self.assertEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
