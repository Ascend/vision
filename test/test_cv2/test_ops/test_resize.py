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
from torchvision import transforms as trans
from test_cv2_utils import image_similarity_vectors_via_cos


@pytest.mark.parametrize(
    ["img_path", "size", "interpolation"],
    [
        ("./test/Data/fish/fish_11.jpg", (1000, 800), 2),
        ("./test/Data/fish/fish_22.jpg", 500, 2),
        ("./test/Data/fish/fish_33.jpg", 200, 2),
        ("./test/Data/fish/fish_44.jpg", (100, 80), 2),
        ("./test/Data/fish/fish_55.jpg", 30, 2),
    ],
)
def test_resize_shape(img_path, size, interpolation):
    pil_img = Image.open(img_path)

    # using pil resize
    pil_resize = trans.Resize(size=size, interpolation=interpolation)(pil_img)

    # using cv2+convert resize
    import torchvision_npu
    torchvision_npu.set_image_backend("cv2")
    cv2_resize = trans.Resize(size=size, interpolation=interpolation)(pil_img)

    assert type(pil_resize) == type(cv2_resize)
    assert pil_resize.size == cv2_resize.size
    assert image_similarity_vectors_via_cos(pil_resize, cv2_resize)


@pytest.mark.parametrize(
    ["img_path", "interpolation"],
    [
        ("./test/Data/fish/fish_11.jpg", 0),
        ("./test/Data/fish/fish_11.jpg", 1),
        ("./test/Data/fish/fish_11.jpg", 2),
        ("./test/Data/fish/fish_11.jpg", 3),
        ("./test/Data/fish/fish_11.jpg", 4),
        ("./test/Data/fish/fish_11.jpg", 5)
    ],
)
def test_resize_interpolation(img_path, interpolation):
    pil_img = Image.open(img_path)

    # using pil resize
    pil_resize = trans.Resize((224, 224), interpolation=interpolation)(pil_img)

    # using cv2+convert resize
    import torchvision_npu
    torchvision_npu.set_image_backend("cv2")
    cv2_resize = trans.Resize((224, 224), interpolation=interpolation)(pil_img)

    assert type(pil_resize) == type(cv2_resize)
    assert pil_resize.size == cv2_resize.size
    assert image_similarity_vectors_via_cos(pil_resize, cv2_resize)


@pytest.mark.parametrize(
    ["img_path", "size", "interpolation"],
    [
        ("./test/Data/fish/fish_11.jpg", (1000, 800), 0),
        ("./test/Data/fish/fish_22.jpg", 500, 1),
        ("./test/Data/fish/fish_33.jpg", 200, 2),
        ("./test/Data/fish/fish_44.jpg", (100, 80), 3),
        ("./test/Data/fish/fish_55.jpg", 30, 4),
        ("./test/Data/fish/fish_55.jpg", 30, 5),
    ],
)
def test_scale(img_path, size, interpolation):
    pil_img = Image.open(img_path)

    # using pil resize
    pil_resize = trans.Resize(size=size, interpolation=interpolation)(pil_img)

    # using cv2+convert resize
    import torchvision_npu
    torchvision_npu.set_image_backend("cv2")
    cv2_resize = trans.Resize(size=size, interpolation=interpolation)(pil_img)

    assert type(pil_resize) == type(cv2_resize)
    assert pil_resize.size == cv2_resize.size
    assert image_similarity_vectors_via_cos(pil_resize, cv2_resize)
