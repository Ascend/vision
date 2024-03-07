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
    ["img_path", "transforms", "p"],
    [
        ("./test/Data/fish/fish_11.jpg",
         [trans.ToTensor(), trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])], 0.1),
        ("./test/Data/fish/fish_22.jpg",
         [trans.RandomResizedCrop(224), trans.RandomHorizontalFlip()], 0.3),
        ("./test/Data/fish/fish_33.jpg",
         [trans.Resize(224), trans.CenterCrop(224)], 0.5),
        ("./test/Data/fish/fish_44.jpg",
         [trans.RandomRotation(60), trans.GaussianBlur(1, 10)], 0.7),
        ("./test/Data/fish/fish_55.jpg",
         [trans.RandomAdjustSharpness(1, 0.7), trans.RandomPerspective(0.5)], 1),
    ],
)
def test_random_apply(img_path, transforms, p):
    pil_img = Image.open(img_path)

    # using pil random appy
    torchvision.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_apply = trans.RandomApply(transforms=transforms, p=p)(pil_img)

    # using cv2 random appy
    torchvision.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_apply = trans.RandomApply(transforms=transforms, p=p)(cv2_img)

    assert isinstance(pil_apply, Image.Image) and isinstance(cv2_apply, np.ndarray)
    assert pil_apply.size == cv2_apply.shape[:2][::-1]
    assert image_similarity_vectors_via_cos(pil_apply, Image.fromarray(cv2_apply))
    
    pil_img.close()
