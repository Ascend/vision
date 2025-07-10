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
import torchvision
from torchvision import transforms as trans
from test_cv2_utils import image_similarity_vectors_via_cos
import torchvision_npu


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
    torchvision.set_image_backend("PIL")
    pil_resize = trans.Resize(size=size, interpolation=interpolation)(pil_img)

    # using cv2 resize
    torchvision.set_image_backend("cv2")
    cv2_img = np.asarray(pil_img)
    cv2_resize = trans.Resize(size=size, interpolation=interpolation)(cv2_img)

    assert isinstance(pil_resize, Image.Image) and isinstance(cv2_resize, np.ndarray)
    assert pil_resize.size == cv2_resize.shape[:2][::-1]
    assert image_similarity_vectors_via_cos(pil_resize, Image.fromarray(cv2_resize))
    
    pil_img.close()


@pytest.mark.parametrize(
    ["img_path", "interpolation"],
    [
        ("./test/Data/fish/fish_22.jpg", 0),
        ("./test/Data/fish/fish_22.jpg", 1),
        ("./test/Data/fish/fish_22.jpg", 2),
        ("./test/Data/fish/fish_22.jpg", 3),
    ],
)
def test_resize_interpolation(img_path, interpolation):
    pil_img = Image.open(img_path)

    # using pil resize
    torchvision.set_image_backend("PIL")
    pil_resize = trans.Resize((224, 224), interpolation=interpolation)(pil_img)

    # using cv2 resize
    torchvision.set_image_backend("cv2")
    cv2_img = np.asarray(pil_img)
    cv2_resize = trans.Resize((224, 224), interpolation=interpolation)(cv2_img)

    assert isinstance(pil_resize, Image.Image) and isinstance(cv2_resize, np.ndarray)
    assert pil_resize.size == cv2_resize.shape[:2][::-1]
    assert image_similarity_vectors_via_cos(pil_resize, Image.fromarray(cv2_resize))
    
    pil_img.close()


@pytest.mark.parametrize(
    ["img_path", "size", "interpolation"],
    [
        ("./test/Data/fish/fish_11.jpg", (1000, 800), 0),
        ("./test/Data/fish/fish_22.jpg", 500, 1),
        ("./test/Data/fish/fish_33.jpg", 200, 2),
        ("./test/Data/fish/fish_44.jpg", (100, 80), 3),
    ],
)
def test_scale(img_path, size, interpolation):
    pil_img = Image.open(img_path)

    # using pil resize
    torchvision.set_image_backend("PIL")
    pil_resize = trans.Resize(size=size, interpolation=interpolation)(pil_img)

    # using cv2 resize
    torchvision.set_image_backend("cv2")
    cv2_img = np.asarray(pil_img)
    cv2_resize = trans.Resize(size=size, interpolation=interpolation)(cv2_img)

    assert isinstance(pil_resize, Image.Image) and isinstance(cv2_resize, np.ndarray)
    assert pil_resize.size == cv2_resize.shape[:2][::-1]
    assert image_similarity_vectors_via_cos(pil_resize, Image.fromarray(cv2_resize))
    
    pil_img.close()


@pytest.mark.parametrize(
    ["img_path", "size", "interpolation"],
    [
        ("./test/Data/fish/fish_11.jpg", (1000, 800), 0),
        ("./test/Data/fish/fish_22.jpg", 500, 1),
        ("./test/Data/fish/fish_33.jpg", 200, 2),
        ("./test/Data/fish/fish_44.jpg", (100, 80), 3),
    ],
)
def test_cv2_PIL_image(img_path, size, interpolation):
    pil_img = Image.open(img_path)

    # using cv2 resize
    torchvision.set_image_backend("cv2")
    # input is np.array
    cv2_img = np.asarray(pil_img)
    array_resize = trans.Resize(size=size, interpolation=interpolation)(cv2_img)
    # input is Image.Image
    image_resize = trans.Resize(size=size, interpolation=interpolation)(pil_img)

    assert isinstance(image_resize, Image.Image) and isinstance(array_resize, np.ndarray)
    assert image_resize.size == array_resize.shape[:2][::-1]
    if interpolation == 0:
        assert image_similarity_vectors_via_cos(image_resize, Image.fromarray(array_resize))
    else:
        assert np.equal(np.array(image_resize), array_resize).all()
    
    pil_img.close()
