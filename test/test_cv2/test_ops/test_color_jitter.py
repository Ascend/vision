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

from PIL import Image
import pytest
import torch
from torchvision import transforms as trans
from test_cv2_utils import image_similarity_vectors_via_cos


@pytest.mark.parametrize(
    ["img_path", "brightness", "contrast", "saturation", "hue"],
    [
        ("./test/Data/fish/fish_11.jpg", [0.1, 0.5], [0.1, 1], 0.5, 0.3),
        ("./test/Data/fish/fish_22.jpg", 0.3, 0.5, 1, [0, 0.5]),
        ("./test/Data/fish/fish_33.jpg", 1, 0.7, (0, 1), [0.1, 0.5]),
        ("./test/Data/fish/fish_44.jpg", 0, 0.3, [0.6, 1], 0),
        ("./test/Data/fish/fish_55.jpg", (0.1, 1), 0.7, (0.1, 0.3), 0.8),
    ],
)
def test_color_jitter(img_path, brightness, contrast, saturation, hue):
    pil_img = Image.open(img_path)

    # using pil color jitter
    torch.manual_seed(10)
    pil_brightness = trans.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)(
        pil_img)

    # using cv2+convert color jitter
    import torchvision_npu
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_brightness = trans.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)(
        pil_img)

    assert type(pil_brightness) == type(cv2_brightness)
    assert pil_brightness.size == cv2_brightness.size
    assert image_similarity_vectors_via_cos(pil_brightness, cv2_brightness)
