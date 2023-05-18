import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision
import torchvision_npu
import torchvision.transforms as transforms

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
        path = "../Data/dog/dog.0001.jpg"
        cpu_input = torchvision.datasets.folder.pil_loader(path)
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)

        cpu_output = self.cpu_op_exec(cpu_input)
        npu_output = self.npu_op_exec(npu_input)

        self.assertEqual(npu_output, cpu_output)


if __name__ == '__main__':
    run_tests()
