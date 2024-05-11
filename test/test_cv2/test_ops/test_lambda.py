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
import torchvision
from torchvision import transforms as trans
import cv2
import torchvision_npu


def custom_crop(img, pos, size):
    ow, oh = img.shape[:2][::-1]
    x1, y1 = pos
    tw = th = size

    if ow > tw or oh > th:
        return cv2.crop((x1, y1, x1 + tw, y1 + th))
    return img


@pytest.mark.parametrize(
    "img_path",
    [
        "./test/Data/fish/fish_11.jpg",
    ],
)
def test_lambda(img_path):
    pil_img = Image.open(img_path)

    # using pil lambda
    torchvision.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_lambda = trans.Lambda(lambda img: custom_crop(pil_img, (5, 5), 224)),

    # using cv2 lambda
    torchvision.set_image_backend("cv2")
    cv2_img = np.asarray(pil_img)
    cv2_lambda = trans.Lambda(lambda img: custom_crop(cv2_img, (5, 5), 224)),

    assert type(pil_lambda) == type(cv2_lambda)
    
    pil_img.close()
