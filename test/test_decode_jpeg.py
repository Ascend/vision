# Copyright (c) 2022, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import torch
import torchvision_npu


class TestDecodeJpeg(unittest.TestCase):
    def test_decode_jpeg(self):
        path = "./Data/dog/dog.0001.jpg"
        out_put = torchvision_npu.datasets.npu_loader(path)
        self.assertEqual(out_put.device.type, 'npu')
        self.assertEqual(out_put.dtype, torch.uint8)
        self.assertEqual(out_put.shape, torch.Size([1, 3, 355, 432]))

    def test_decode_bmp(self):
        path = "./Data/bmp/pic.bmp"
        out_put = torchvision_npu.npu_loader(path)
        self.assertEqual(out_put.device.type, 'npu')
        self.assertEqual(out_put.dtype, torch.uint8)
        self.assertEqual(out_put.shape, torch.Size([1, 3, 360, 330]))



if __name__ == '__main__':
    unittest.main()
