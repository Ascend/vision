import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision_npu
import torchvision.transforms as transforms

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestCrop(TestCase):
    def test_center_crop(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        cpu_output = transforms.CenterCrop((100, 200))(cpu_input)
        npu_output = transforms.CenterCrop((100, 200))(npu_input).cpu().squeeze(0)
        self.assertEqual(cpu_output, npu_output)

    def test_five_crop(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        cpu_output = transforms.FiveCrop((100))(cpu_input)
        npu_output = transforms.FiveCrop((100))(npu_input)
        for i in range(5):
            npu_output_i = npu_output[i].cpu().squeeze(0)
            self.assertEqual(cpu_output[i], npu_output_i)

    def test_ten_crop(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        for is_vflip in [False, True]:
            cpu_output = transforms.TenCrop(50, is_vflip)(cpu_input)
            npu_output = transforms.TenCrop(50, is_vflip)(npu_input)
            for i in range(10):
                npu_output_i = npu_output[i].cpu().squeeze(0)
                self.assertEqual(cpu_output[i], npu_output_i)


if __name__ == '__main__':
    run_tests()
