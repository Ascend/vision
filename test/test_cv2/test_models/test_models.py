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

import argparse

import PIL.Image
import numpy as np
import os
import psutil
import setproctitle

import time
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.nn.functional as F
from torchvision import models, transforms
import torchvision.datasets as dset
from torch.nn.parallel import DistributedDataParallel as DDP
from apex import amp
import torchvision_npu

torch.npu.set_compile_mode(jit_compile=False)

IMG_RESIZE = 224
IMAGENET_DATASET_PATH = './imagenet/train'  # change to your imagenet path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--nEpoch', type=int, default=3)
    parser.add_argument('--seed', type=int, default=49)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--world_size', type=int, default=8)
    parser.add_argument('--local_rank', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--backend', type=str, default='PIL')
    parser.add_argument('--distribute_func', type=str, default='launch')

    args = parser.parse_args()
    torchvision_npu.set_image_backend(args.backend)
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    torch.npu.set_compile_mode(jit_compile=False)
    if args.distribute_func == 'launch':
        main_worker(args=args)
    elif args.distribute_func == 'spawn':
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))


def get_transforms():
    base_trans = [
        transforms.RandomResizedCrop(IMG_RESIZE),
        transforms.RandomHorizontalFlip(),
    ]
    return transforms.Compose(base_trans)


def main_worker(rank_id=-1, args=None):
    if args.distribute_func == 'spawn':
        args.local_rank = rank_id

    p = psutil.Process()
    cpu_list = p.cpu_affinity()
    core_per_proc = len(cpu_list) // args.world_size
    p.cpu_affinity(cpu_list[args.local_rank * core_per_proc:(args.local_rank + 1) * core_per_proc])
    print('============== use core:{}'.format(
        cpu_list[args.local_rank * core_per_proc:(args.local_rank + 1) * core_per_proc]
    ))
    process = '{}_{}_{}'.format(args.network, args.backend, args.local_rank)
    print('============= process:{}'.format(process))
    setproctitle.setproctitle(process)
    if args.world_size > 1:
        dist.init_process_group(backend='hccl', world_size=args.world_size, rank=args.local_rank)

    if args.network == 'resnet18':
        net = models.resnet18(num_classes=1000)
    elif args.network == 'resnet50':
        net = models.resnet50(num_classes=1000)
    elif args.network == 'mobilenetv2':
        net = models.MobileNetV2(num_classes=1000)

    loc = 'npu:{}'.format(args.local_rank)
    torch.npu.set_device(loc)
    net = net.to(loc)

    if args.opt == 'sgd':
        optimizer = build_SGD(
            parameters=list(net.named_parameters()),
            lr=1.6, momentum=0.9, weight_decay=0.0001
        )
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    net, optimizer = amp.initialize(net, optimizer, opt_level='O2', loss_scale=1024, verbosity=1)
    if args.worldsize > 1:
        net = DDP(net, device_ids=[args.local_rank], broadcast_buffer=False)

    train_transforms = get_transforms()
    if args.dataset == 'imagenet':
        dataset = dset.ImageFolder(
            loader=torchvision_npu.dataset.folder._cv2_loader,
            root=IMAGENET_DATASET_PATH,
            transform=train_transforms,
        )
    print('===================== dist')
    dataloader_fn = MultiEpochsDataLoader
    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        train_loader = dataloader_fn(
            dataset, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True,
            sampler=train_sampler, drop_last=True,
            collate_fn=fast_collate
        )
    else:
        train_loader = dataloader_fn(
            dataset, batch_size=args.batch_size,
            num_workers=args.num_workers, shuffle=False,
            drop_last=True, pin_memory=False,
            collate_fn=fast_collate
        )
    print('================== loop')

    for epoch in range(args.nEpochs):
        if args.world_size > 1:
            train_sampler.set_epoch(epoch)
        train(args, epoch, net, train_loader, optimizer)


def train(args, epoch, net, train_loader, optimizer):
    net.train()
    local_rank = 'npu:{}'.format(args.local_rank)

    e2e_begin = time.time()
    count = 50

    for i, (img, target) in enumerate(train_loader):

        img = img.to(local_rank, non_blocking=True).to(torch.float)
        target = target.to(torch.int32).to(local_rank, non_blocking=True)

        output = net(img)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()

        if i % count == 0 and not i == 0 and args.local_rank == 0:
            print('epoch:{}, step:{}, fps:{}'.format(epoch, i, args.batch_size * args.world_size * count / (
                    time.time() - e2e_begin)))
            e2e_begin = time.time()


def build_SGD(parameters, lr, momentum, weight_decay, nesterov=False):
    bn_params = [v for n, v in parameters if 'bn' in n]
    rest_params = [v for n, v in parameters if not 'bn' in n]
    optimizer = torch.optim.SGD(
        [{'params': bn_params, 'weight_decay': 0},
         {'params': rest_params, 'weight_decay': weight_decay}],
        lr, monmentum=momentum, weight_decay=weight_decay, nesterov=nesterov
    )
    return optimizer


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """
    sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    if isinstance(imgs[0], PIL.Image.Image):
        w = imgs[0].size[0]
        h = imgs[0].size[1]
    else:
        w = imgs[0].shape[1]
        h = imgs[0].shape[0]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=torch.contiguous_format
    )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array.copy())
    return tensor, targets


if __name__ == "__main__":
    main()
