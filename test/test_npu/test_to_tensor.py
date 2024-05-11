import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision
import torchvision.transforms as transforms
import torchvision_npu

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestToTensor(TestCase):
    @staticmethod
    def cpu_op_exec(input1):
        output = transforms.ToTensor()(input1)
        output = output.numpy()
        return output

    @staticmethod
    def npu_op_exec(input1):
        output = transforms.ToTensor()(input1)
        output = output.cpu().squeeze(0)
        output = output.numpy()
        return output

    def test_to_tensor(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        cpu_input = torchvision.datasets.folder.pil_loader(path)
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)

        cpu_output = self.cpu_op_exec(cpu_input)

        torch.npu.set_compile_mode(jit_compile=True)
        npu_output = self.npu_op_exec(npu_input)
        self.assertEqual(npu_output, cpu_output)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = self.npu_op_exec(npu_input)
        self.assertEqual(npu_output, cpu_output)


if __name__ == '__main__':
    run_tests()
