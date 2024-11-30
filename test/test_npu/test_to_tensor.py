import os
from pathlib import Path
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision
import torchvision.transforms as transforms
import torchvision_npu


torch_npu.npu.current_stream().set_data_preprocess_stream(True)
TEST_DIR = Path(__file__).resolve().parents[1]


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

        path = os.path.join(TEST_DIR, "Data/dog/dog.0001.jpg")
        cpu_input = torchvision.datasets.folder.pil_loader(path)
        npu_input = torchvision_npu.datasets._folder._npu_loader(path)

        cpu_output = self.cpu_op_exec(cpu_input)

        torch.npu.set_compile_mode(jit_compile=True)
        npu_output = self.npu_op_exec(npu_input)
        self.assertEqual(npu_output, cpu_output)

        torch.npu.set_compile_mode(jit_compile=False)
        npu_output = self.npu_op_exec(npu_input)
        self.assertEqual(npu_output, cpu_output)

    def test_to_tensor_batch(self):
        torch.ops.torchvision._dvpp_init()
        cpu_input = torch.randint(0, 255, (4, 3, 320, 240), dtype=torch.uint8)
        npu_input = cpu_input.npu()

        cpu_output = cpu_input / 255.0
        npu_output = self.npu_op_exec(npu_input)
        self.assertEqual(npu_output, cpu_output)


if __name__ == '__main__':
    run_tests()
