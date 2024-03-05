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
from test_cv2_utils import image_similarity_vectors_via_cos
import torchvision_npu


@pytest.mark.parametrize(
    ["img_path", "degree", "interpolation", "expand", "center", "fill"],
    [
        ("./test/Data/fish/fish_11.jpg", 45, 0, False, None, 0),
        ("./test/Data/fish/fish_22.jpg", 90, 0, False, None, 0),
        ("./test/Data/fish/fish_33.jpg", 180, 0, False, None, 0),
        ("./test/Data/fish/fish_44.jpg", 45, 2, False, None, 0),
        ("./test/Data/fish/fish_55.jpg", 45, 3, False, (45, 45), 0),
        ("./test/Data/fish/fish_55.jpg", 45, 0, False, (45, 45), (255, 0, 0)),

    ],
)
def test_rotate_degree(img_path, degree, interpolation, expand, center, fill):
    pil_img = Image.open(img_path)

    # using pil rotate
    torchvision_npu.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_rotate = trans.RandomRotation(degrees=degree, interpolation=interpolation, expand=expand, center=center,
                                      fill=fill)(pil_img)

    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    # using cv2 rotate
    cv2_rotate = trans.RandomRotation(degrees=degree, interpolation=interpolation, expand=expand, center=center,
                                      fill=fill)(cv2_img)

    assert isinstance(pil_rotate, Image.Image) and isinstance(cv2_rotate, np.ndarray)
    assert pil_rotate.size == cv2_rotate.shape[:2][::-1]
    assert image_similarity_vectors_via_cos(pil_rotate, Image.fromarray(cv2_rotate))
    
    pil_img.close()


@pytest.mark.parametrize(
    ["img_path", "interpolation"],
    [
        ("./test/Data/fish/fish_11.jpg", 0),
        ("./test/Data/fish/fish_11.jpg", 2),
        ("./test/Data/fish/fish_44.jpg", 3)
    ],
)
def test_rotate_interpolation(img_path, interpolation):
    pil_img = Image.open(img_path)

    # using pil rotate
    torchvision_npu.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_rotate = trans.RandomRotation(45, interpolation=interpolation)(pil_img)

    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    # using cv2 rotate
    cv2_rotate = trans.RandomRotation(45, interpolation=interpolation)(cv2_img)

    assert isinstance(pil_rotate, Image.Image) and isinstance(cv2_rotate, np.ndarray)
    assert pil_rotate.size == cv2_rotate.shape[:2][::-1]
    assert image_similarity_vectors_via_cos(pil_rotate, Image.fromarray(cv2_rotate))
    
    pil_img.close()
