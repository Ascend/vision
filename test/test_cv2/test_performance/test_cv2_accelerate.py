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
import time

import numpy as np
import torch
import torchvision.datasets
from torchvision import transforms
from PIL import Image

import torchvision_npu


IMAGENET_PATH = "./test/Data/"
IMAGE_SIZE = 224
EPOCH = 10
BATCH_SIZE = 256
IMAGE_RESIZE = 256


class ImagenetHandle(object):
    def __init__(self, imagenet_path):
        self.imagenet_path = imagenet_path
        self.img_size = IMAGE_SIZE
        self.epoch = EPOCH
        self.batch_size = BATCH_SIZE
        self.img_resize = IMAGE_RESIZE

    def imagenet_valid_dataset(self):
        transform = [
            transforms.Resize(self.img_resize),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ]
        dataset = torchvision.datasets.ImageFolder(self.imagenet_path, transforms.Compose(transform))
        return dataset

    def imagenet_train_dataset(self):
        transform = [
            transforms.RandomResizedCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ]
        dataset = torchvision.datasets.ImageFolder(self.imagenet_path, transforms.Compose(transform))
        return dataset

    def trans_dataset(self):
        train_loader = self.imagenet_train_dataset()
        valid_loader = self.imagenet_valid_dataset()
        train_fps = 0
        valid_fps = 0
        for index in range(self.epoch):
            train_begin_time = time.time()
            for batch_idx, (imgs, target) in enumerate(train_loader):
                continue
            train_fps += len(train_loader) / (time.time() - train_begin_time)

            valid_begin_time = time.time()
            for batch_idx, (imgs, target) in enumerate(valid_loader):
                continue
            valid_fps += len(valid_loader) / (time.time() - valid_begin_time)

        print('train data {:.4f} FPS , valid data {:.4f} FPS'.format(train_fps / self.epoch, valid_fps / self.epoch))

        return train_fps / self.epoch, valid_fps / self.epoch


def test_cv2_accelerate():
    torchvision.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_train_fps, pil_valid_fps = ImagenetHandle(IMAGENET_PATH).trans_dataset()

    torchvision.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_train_fps, cv2_valid_fps = ImagenetHandle(IMAGENET_PATH).trans_dataset()
    assert pil_train_fps < cv2_train_fps
