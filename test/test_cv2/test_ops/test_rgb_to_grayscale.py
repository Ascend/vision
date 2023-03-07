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
    ["img_path", "num_out_channels"],
    [
        ("./test/Data/fish/fish_11.jpg", 1),
        ("./test/Data/fish/fish_22.jpg", 3),
        ("./test/Data/fish/fish_33.jpg", 3),
        ("./test/Data/fish/fish_44.jpg", 3),
        ("./test/Data/fish/fish_55.jpg", 3),
    ],
)
def test_grayscale(img_path, num_out_channels):
    pil_img = Image.open(img_path)

    # using pil grayscale
    torchvision_npu.set_image_backend("PIL")
    pil_grayscale = trans.Grayscale(num_output_channels=num_out_channels)(pil_img)

    # using cv2 grayscale
    torchvision_npu.set_image_backend("cv2")
    cv2_img = np.asarray(pil_img)
    cv2_grayscale = trans.Grayscale(num_output_channels=num_out_channels)(cv2_img)

    assert isinstance(pil_grayscale, Image.Image) and isinstance(cv2_grayscale, np.ndarray)
    assert pil_grayscale.size == cv2_grayscale.shape[:2][::-1]
    assert image_similarity_vectors_via_cos(pil_grayscale, Image.fromarray(cv2_grayscale))


@pytest.mark.parametrize(
    ["img_path", "num_out_channels"],
    [
        ("./test/Data/fish/fish_11.jpg", 1),
        ("./test/Data/fish/fish_22.jpg", 3),
        ("./test/Data/fish/fish_33.jpg", 3),
        ("./test/Data/fish/fish_44.jpg", 3),
        ("./test/Data/fish/fish_55.jpg", 3),
    ],
)
def test_random_grayscale(img_path, num_out_channels):
    pil_img = Image.open(img_path)

    # using pil grayscale
    torchvision_npu.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_grayscale = trans.Grayscale(num_output_channels=num_out_channels)(pil_img)

    # using cv2 grayscale
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_grayscale = trans.Grayscale(num_output_channels=num_out_channels)(cv2_img)

    assert isinstance(pil_grayscale, Image.Image) and isinstance(cv2_grayscale, np.ndarray)
    assert pil_grayscale.size == cv2_grayscale.shape[:2][::-1]
    assert image_similarity_vectors_via_cos(pil_grayscale, Image.fromarray(cv2_grayscale))
