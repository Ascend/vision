import torch
import torch_npu
from torch_npu.testing.testcase import run_tests, TestCase
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
import torchvision
from torchvision import ops
import torchvision_npu


class TestNms(TestCase):
    @skipIfUnsupportMultiNPU(2)
    def test_nms_multidevice(self):
        boxes = torch.tensor([[285.3538, 185.5758, 1193.5110, 851.4551],
                              [285.1472, 188.7374, 1192.4984, 851.0669],
                              [279.2440, 197.9812, 1189.4746, 849.2019]]).to('npu:1')
        scores = torch.tensor([0.6370, 0.7569, 0.3966]).to('npu:1')
        iou_thres = 0.2
        cpu_res = ops.nms(boxes.cpu(), scores.cpu(), iou_thres)
        npu_res = ops.nms(boxes, scores, iou_thres)
        self.assertRtolEqual(cpu_res, npu_res)


if __name__ == '__main__':
    run_tests()
