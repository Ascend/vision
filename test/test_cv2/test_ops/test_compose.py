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
import numpy as np
import torch
import torchvision
from torchvision import transforms as trans
from test_cv2_utils import image_similarity_vectors_via_cos
import torchvision_npu


@pytest.mark.parametrize(
    ["img_path", "transforms"],
    [
        ("./test/Data/fish/fish_11.jpg",
         [trans.ToTensor(), trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        ("./test/Data/fish/fish_22.jpg",
         [trans.RandomResizedCrop(224), trans.RandomHorizontalFlip()]),
        ("./test/Data/fish/fish_33.jpg",
         [trans.Resize(224), trans.CenterCrop(224)]),
        ("./test/Data/fish/fish_44.jpg",
         [trans.RandomRotation(60), trans.GaussianBlur(1, 10)]),
        ("./test/Data/fish/fish_55.jpg",
         [trans.RandomAdjustSharpness(1, 0.7), trans.RandomPerspective(0.5)]),
    ],
)
def test_compose(img_path, transforms):
    pil_img = Image.open(img_path)

    # using pil compose
    torchvision.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_compose = trans.Compose(transforms=transforms)(
        pil_img)

    # using cv2 compose
    torchvision.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_compose = trans.Compose(transforms=transforms)(cv2_img)
    if isinstance(pil_compose, torch.Tensor):
        pil_compose = trans.ToPILImage()(pil_compose)
        cv2_compose = trans.ToPILImage()(cv2_compose)
        assert pil_compose.size == cv2_compose.size
        assert (np.array(pil_compose) == np.array(cv2_compose)).all()
    else:
        assert isinstance(pil_compose, Image.Image) and isinstance(cv2_compose, np.ndarray)
        assert pil_compose.size == cv2_compose.shape[:2][::-1]
        assert image_similarity_vectors_via_cos(pil_compose, Image.fromarray(cv2_compose))
    
    pil_img.close()
