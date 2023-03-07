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

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms as trans
import torchvision_npu


@pytest.mark.parametrize(
    ["img_path", "padding", "fill", "padding_mode"],
    [
        ("./test/Data/fish/fish_11.jpg", 100, 0, "constant"),
        ("./test/Data/fish/fish_22.jpg", (100, 200), 0, "constant"),
        ("./test/Data/fish/fish_33.jpg", (10, 20, 30, 40), (255, 0, 0), "constant"),
        ("./test/Data/fish/fish_44.jpg", (10, 20, 30, 40), (255, 0, 0), "edge"),
        ("./test/Data/fish/fish_55.jpg", (10, 20, 30, 40), (255, 0, 0), "reflect"),
        ("./test/Data/fish/fish_33.jpg", (10, 20, 30, 40), (255, 0, 0), "symmetric")
    ],
)
def test_pad(img_path, padding, fill, padding_mode):
    pil_img = Image.open(img_path)

    # using pil pad
    torchvision_npu.set_image_backend("PIL")
    pil_pad = trans.Pad(padding=padding, fill=fill, padding_mode=padding_mode)(pil_img)

    # using cv2 pad
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_pad = trans.Pad(padding=padding, fill=fill, padding_mode=padding_mode)(cv2_img)

    assert isinstance(pil_pad, Image.Image) and isinstance(cv2_pad, np.ndarray)
    assert pil_pad.size == cv2_pad.shape[:2][::-1]
    assert (np.array(pil_pad) == cv2_pad).all()
