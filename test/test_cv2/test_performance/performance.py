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
import timeit

import numpy as np
import torch
import cv2
import torchvision.datasets
from PIL import Image
from torchvision import transforms

import torchvision_npu


import_module = "from torchvision import transforms; " \
                "from __main__ import pil_img, cv2_img, test_data, pil_load, cv2_load, pil_trans_np"

test_samples = {
    "Resize": {
        "pil_ops": "transforms.Resize((224,224))(pil_img)",
        "cv2_ops": "transforms.Resize((224,224))(cv2_img)",
    },
    "Scale": {
        "pil_ops": "transforms.Scale((224,224))(pil_img)",
        "cv2_ops": "transforms.Scale((224,224))(cv2_img)",
    },
    "PILToTensor": {
        "pil_ops": "transforms.PILToTensor()(pil_img)",
        "cv2_ops": "transforms.PILToTensor()(cv2_img)",
    },
    "ToTensor": {
        "pil_ops": "transforms.ToTensor()(pil_img)",
        "cv2_ops": "transforms.ToTensor()(cv2_img)",
    },
    "CenterCrop": {
        "pil_ops": "transforms.CenterCrop((224,224))(pil_img)",
        "cv2_ops": "transforms.CenterCrop((224,224))(cv2_img)",
    },
    "Pad": {
        "pil_ops": "transforms.Pad([224,224],fill=0,padding_mode='constant')(pil_img)",
        "cv2_ops": "transforms.Pad([224,224],fill=0,padding_mode='constant')(cv2_img)",
    },
    "RandomCrop": {
        "pil_ops": "transforms.RandomCrop((224,224))(pil_img)",
        "cv2_ops": "transforms.RandomCrop((224,224))(cv2_img)",
    },
    "RandomHorizontalFlip": {
        "pil_ops": "transforms.RandomHorizontalFlip(p=0.5)(pil_img)",
        "cv2_ops": "transforms.RandomHorizontalFlip(p=0.5)(cv2_img)",
    },
    "RandomVerticalFlip": {
        "pil_ops": "transforms.RandomVerticalFlip(p=0.5)(pil_img)",
        "cv2_ops": "transforms.RandomVerticalFlip(p=0.5)(cv2_img)",
    },
    "RandomResizedCrop": {
        "pil_ops": "transforms.RandomResizedCrop((224,224))(pil_img)",
        "cv2_ops": "transforms.RandomResizedCrop((224,224))(cv2_img)",
    },
    "FiveCrop": {
        "pil_ops": "transforms.FiveCrop(32)(pil_img)",
        "cv2_ops": "transforms.FiveCrop(32)(cv2_img)",
    },
    "TenCrop": {
        "pil_ops": "transforms.TenCrop(32)(pil_img)",
        "cv2_ops": "transforms.TenCrop(32)(cv2_img)",
    },
    "ColorJitter_bright": {
        "pil_ops": "transforms.ColorJitter(brightness=[0.01,0.05])(pil_img)",
        "cv2_ops": "transforms.ColorJitter(brightness=[0.01,0.05])(cv2_img)",
    },
    "ColorJitter_contrast": {
        "pil_ops": "transforms.ColorJitter(contrast=[0.3,0.6])(pil_img)",
        "cv2_ops": "transforms.ColorJitter(contrast=[0.3,0.6])(cv2_img)",
    },
    "ColorJitter_saturation": {
        "pil_ops": "transforms.ColorJitter(saturation=0.5)(pil_img)",
        "cv2_ops": "transforms.ColorJitter(saturation=0.5)(cv2_img)",
    },
    "ColorJitter_hue": {
        "pil_ops": "transforms.ColorJitter(hue=0.5)(pil_img)",
        "cv2_ops": "transforms.ColorJitter(hue=0.5)(cv2_img)",
    },
    "RandomRotation": {
        "pil_ops": "transforms.RandomRotation(45)(pil_img)",
        "cv2_ops": "transforms.RandomRotation(45)(cv2_img)",
    },
    "RandomAffine": {
        "pil_ops": "transforms.RandomAffine(degrees=(10,150))(pil_img)",
        "cv2_ops": "transforms.RandomAffine(degrees=(10,150))(cv2_img)",
    },
    "Grayscale": {
        "pil_ops": "transforms.Grayscale(3)(pil_img)",
        "cv2_ops": "transforms.Grayscale(3)(cv2_img)",
    },
    "RandomGrayscale": {
        "pil_ops": "transforms.RandomGrayscale(p=0.8)(pil_img)",
        "cv2_ops": "transforms.RandomGrayscale(p=0.8)(cv2_img)",
    },
    "RandomPerspective": {
        "pil_ops": "transforms.RandomPerspective(distortion_scale=0.5, p=0.5)(pil_img)",
        "cv2_ops": "transforms.RandomPerspective(distortion_scale=0.5, p=0.5)(cv2_img)",
    },
    "GaussianBlur": {
        "pil_ops": "transforms.GaussianBlur(11,100)(pil_img)",
        "cv2_ops": "transforms.GaussianBlur(11,100)(cv2_img)",
    },
    "RandomInvert": {
        "pil_ops": "transforms.RandomInvert(p=0.5)(pil_img)",
        "cv2_ops": "transforms.RandomInvert(p=0.5)(cv2_img)",
    },
    "RandomPosterize": {
        "pil_ops": "transforms.RandomPosterize(5)(pil_img)",
        "cv2_ops": "transforms.RandomPosterize(5)(cv2_img)",
    },
    "RandomSolarize": {
        "pil_ops": "transforms.RandomSolarize(100)(pil_img)",
        "cv2_ops": "transforms.RandomSolarize(100)(cv2_img)",
    },
    "RandomAdjustSharpness": {
        "pil_ops": "transforms.RandomAdjustSharpness(2)(pil_img)",
        "cv2_ops": "transforms.RandomAdjustSharpness(2)(cv2_img)",
    },
    "RandomAutocontrast": {
        "pil_ops": "transforms.RandomAutocontrast()(pil_img)",
        "cv2_ops": "transforms.RandomAutocontrast()(cv2_img)",
    },
    "RandomEqualize": {
        "pil_ops": "transforms.RandomEqualize()(pil_img)",
        "cv2_ops": "transforms.RandomEqualize()(cv2_img)",
    },
    "Compose": {
        "pil_ops": "transforms.Compose([transforms.ToTensor(),"
                   "transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(pil_img)",
        "cv2_ops": "transforms.Compose([transforms.ToTensor(),"
                   "transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(cv2_img)",
    },
    "RandomApply": {
        "pil_ops": "transforms.RandomApply([transforms.Resize((224,224)), transforms.RandomAutocontrast()])(pil_img)",
        "cv2_ops": "transforms.RandomApply([transforms.Resize((224,224)), transforms.RandomAutocontrast()])(cv2_img)",
    },
    "RandomChoice": {
        "pil_ops": "transforms.RandomChoice([transforms.Resize((224,224)), transforms.Grayscale(3)])(pil_img)",
        "cv2_ops": "transforms.RandomChoice([transforms.Resize((224,224)), transforms.Grayscale(3)])(cv2_img)",
    },
    "RandomOrder": {
        "pil_ops": "transforms.RandomOrder([transforms.Resize((224,224)), transforms.Grayscale(3)])(pil_img)",
        "cv2_ops": "transforms.RandomOrder([transforms.Resize((224,224)), transforms.Grayscale(3)])(cv2_img)",
    },

}

test_data = "./test/Data/dog/dog.0001.jpg"


def pil_ops_performance(pil_ops, loop):
    torchvision_npu.set_image_backend("PIL")
    pil_handle = timeit.Timer(stmt=pil_ops, setup=import_module)
    pil_ops_spend = pil_handle.timeit(number=loop)
    return pil_ops_spend


def cv2_ops_performance(cv2_ops, loop):
    torchvision_npu.set_image_backend("cv2")
    cv2_handle = timeit.Timer(stmt=cv2_ops, setup=import_module)
    cv2_ops_spend = cv2_handle.timeit(number=loop)
    return cv2_ops_spend


def get_performance(pil_ops: str, cv2_ops: str, loop: int = 1000):
    pil_ops_spend = pil_ops_performance(pil_ops, loop)
    cv2_ops_spend = cv2_ops_performance(cv2_ops, loop)

    return pil_ops_spend, cv2_ops_spend


def compare_cv2_pil_ops():
    performance_better_backend = {"pil": [], "cv2": []}
    for ops_name, sample in test_samples.items():
        pil_ops_spend, cv2_ops_spend = get_performance(**sample)

        print("ops: {} pil: {:.4f} cv2: {:.4f}".format(
            ops_name, pil_ops_spend, cv2_ops_spend
        ))

        if pil_ops_spend < cv2_ops_spend:
            performance_better_backend['pil'].append(ops_name)
        else:
            performance_better_backend['cv2'].append(ops_name)

    print("-" * 50)
    print("performace better: pil {}, cv2_with_trans {}".format(performance_better_backend['pil'],
                                                                performance_better_backend['cv2']))


def pil_load():
    img = Image.open(test_data)
    img_rgb = img.convert('RGB')
    img_rgb = transforms.Resize((224, 224))(img_rgb)
    img.close()
    return img_rgb


def cv2_load():
    img = cv2.imread(test_data)
    img = img[:, :, ::-1]
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    return img


def pil_trans_np():
    img = Image.open(test_data)
    img_rgb = img.convert('RGB')
    img_rgb = np.asarray(img_rgb)
    img_rgb = cv2.resize(img_rgb, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    img.close()
    return img_rgb


def compare_cv2_pil_load():
    pil_load_time = timeit.repeat('pil_load()', setup=import_module, number=1000, repeat=5)
    cv2_load_time = timeit.repeat('cv2_load()', setup=import_module, number=1000, repeat=5)
    pil_trans_time = timeit.repeat('pil_trans_np()', setup=import_module, number=1000, repeat=5)

    print('min time for pil loader', min(pil_load_time))
    print('min time for cv2 loader', min(cv2_load_time))
    print('min time for pil trans loader', min(pil_trans_time))

    if min(cv2_load_time) < min(pil_trans_time):
        raise RuntimeError("cv2 load time shorter than pil trans time.")


if __name__ == "__main__":
    compare_cv2_pil_load()
    compare_cv2_pil_ops()
