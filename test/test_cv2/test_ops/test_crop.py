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
    ["img_path", "size", "padding", "pad_if_need", "fill", "padding_mode"],
    [
        ("./test/Data/fish/fish_11.jpg", 512, None, False, 0, "constant"),
        ("./test/Data/fish/fish_22.jpg", (100, 256), 10, True, 0, "constant"),
        ("./test/Data/fish/fish_33.jpg", (128, 128), 10, False, (255, 0, 0), "constant"),
        ("./test/Data/fish/fish_44.jpg", 32, 10, True, 0, "edge"),
        ("./test/Data/fish/fish_55.jpg", (20, 30), 10, True, 0, "reflect"),
        ("./test/Data/fish/fish_55.jpg", 128, 100, True, 0, "symmetric"),
    ],
)
def test_crop(img_path, size, padding, pad_if_need, fill, padding_mode):
    pil_img = Image.open(img_path)

    # using pil crop
    torchvision_npu.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_crop = trans.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_need, fill=fill,
                                padding_mode=padding_mode)(pil_img)

    # using cv2 crop
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_crop = trans.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_need, fill=fill,
                                padding_mode=padding_mode)(cv2_img)

    assert isinstance(pil_crop, Image.Image) and isinstance(cv2_crop, np.ndarray)
    assert pil_crop.size == cv2_crop.shape[:2][::-1]
    assert (np.array(pil_crop) == cv2_crop).all()
    
    pil_img.close()


@pytest.mark.parametrize(
    ["img_path", "size"],
    [
        ("./test/Data/fish/fish_11.jpg", 512),
        ("./test/Data/fish/fish_22.jpg", (500, 250)),
        ("./test/Data/fish/fish_33.jpg", 200),
        ("./test/Data/fish/fish_44.jpg", (100, 50)),
        ("./test/Data/fish/fish_55.jpg", 30),
    ],
)
def test_center_crop(img_path, size):
    pil_img = Image.open(img_path)

    # using pil center crop
    torchvision_npu.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_center_crop = trans.CenterCrop(size=size)(pil_img)

    # using cv2 center crop
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_center_crop = trans.CenterCrop(size=size)(cv2_img)

    assert isinstance(pil_center_crop, Image.Image) and isinstance(cv2_center_crop, np.ndarray)
    assert pil_center_crop.size == cv2_center_crop.shape[:2][::-1]
    assert (np.array(pil_center_crop) == cv2_center_crop).all()
    
    pil_img.close()


@pytest.mark.parametrize(
    ["img_path", "size", "scale", "ratio", "interpolation"],
    [
        ("./test/Data/fish/fish_11.jpg", 512, (0.08, 1.0), (3. / 4, 4. / 3), 0),
        ("./test/Data/fish/fish_22.jpg", (500, 250), (0.8, 1.0), (3. / 4, 4. / 3), 0),
        ("./test/Data/fish/fish_33.jpg", 300, (0.8, 1.0), (3. / 4, 4. / 3), 1),
        ("./test/Data/fish/fish_44.jpg", (100, 50), (0.08, 1.0), (3. / 4, 4. / 3), 2),
        ("./test/Data/fish/fish_55.jpg", 30, (0.08, 1.0), (3. / 4, 4. / 3), 3),
    ],
)
def test_resized_crop(img_path, size, scale, ratio, interpolation):
    pil_img = Image.open(img_path)

    # using pil center crop
    torchvision_npu.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_resized_crop = trans.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation)(
        pil_img)

    # using cv2 center crop
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_resized_crop = trans.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation)(
        cv2_img)

    assert isinstance(pil_resized_crop, Image.Image) and isinstance(cv2_resized_crop, np.ndarray)
    assert pil_resized_crop.size == cv2_resized_crop.shape[:2][::-1]
    assert image_similarity_vectors_via_cos(pil_resized_crop, Image.fromarray(cv2_resized_crop))
    
    pil_img.close()


@pytest.mark.parametrize(
    ["img_path", "size", "scale", "ratio", "interpolation"],
    [
        ("./test/Data/fish/fish_11.jpg", 512, (0.08, 1.0), (3. / 4, 4. / 3), 0),
        ("./test/Data/fish/fish_22.jpg", (500, 250), (0.8, 1.0), (3. / 4, 4. / 3), 0),
        ("./test/Data/fish/fish_33.jpg", 300, (0.8, 1.0), (3. / 4, 4. / 3), 1),
        ("./test/Data/fish/fish_44.jpg", (100, 50), (0.08, 1.0), (3. / 4, 4. / 3), 2),
        ("./test/Data/fish/fish_55.jpg", 30, (0.08, 1.0), (3. / 4, 4. / 3), 3),
    ],
)
def test_sized_crop(img_path, size, scale, ratio, interpolation):
    pil_img = Image.open(img_path)

    # using pil center crop
    torchvision_npu.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_resized_crop = trans.RandomSizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation)(
        pil_img)

    # using cv2 center crop
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_resized_crop = trans.RandomSizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation)(
        cv2_img)

    assert isinstance(pil_resized_crop, Image.Image) and isinstance(cv2_resized_crop, np.ndarray)
    assert pil_resized_crop.size == cv2_resized_crop.shape[:2][::-1]
    assert image_similarity_vectors_via_cos(pil_resized_crop, Image.fromarray(cv2_resized_crop))
    
    pil_img.close()


@pytest.mark.parametrize(
    ["img_path", "size"],
    [
        ("./test/Data/fish/fish_11.jpg", 512),
        ("./test/Data/fish/fish_22.jpg", (500, 250)),
        ("./test/Data/fish/fish_33.jpg", 200),
        ("./test/Data/fish/fish_44.jpg", (100, 50)),
        ("./test/Data/fish/fish_55.jpg", 30),
    ],
)
def test_five_crop(img_path, size):
    pil_img = Image.open(img_path)

    # using pil five crop
    torchvision_npu.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_five_crop = trans.FiveCrop(size=size)(pil_img)

    # using cv2 five crop
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_five_crop = trans.FiveCrop(size=size)(cv2_img)

    for pil_crop_img, cv2_crop_img in zip(pil_five_crop, cv2_five_crop):
        assert isinstance(pil_crop_img, Image.Image) and isinstance(cv2_crop_img, np.ndarray)
        assert pil_crop_img.size == cv2_crop_img.shape[:2][::-1]
        assert (np.array(pil_crop_img) == cv2_crop_img).all()
    
    pil_img.close()


@pytest.mark.parametrize(
    ["img_path", "size"],
    [
        ("./test/Data/fish/fish_11.jpg", 512),
        ("./test/Data/fish/fish_22.jpg", (500, 250)),
        ("./test/Data/fish/fish_33.jpg", 200),
        ("./test/Data/fish/fish_44.jpg", (100, 50)),
        ("./test/Data/fish/fish_55.jpg", 30),
    ],
)
def test_ten_crop(img_path, size):
    pil_img = Image.open(img_path)

    # using pil ten crop
    torchvision_npu.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_ten_crop = trans.TenCrop(size=size)(pil_img)

    # using cv2 ten crop
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_ten_crop = trans.TenCrop(size=size)(cv2_img)

    for pil_crop_img, cv2_crop_img in zip(pil_ten_crop, cv2_ten_crop):
        assert isinstance(pil_crop_img, Image.Image) and isinstance(cv2_crop_img, np.ndarray)
        assert pil_crop_img.size == cv2_crop_img.shape[:2][::-1]
        assert (np.array(pil_crop_img) == cv2_crop_img).all()
    
    pil_img.close()
