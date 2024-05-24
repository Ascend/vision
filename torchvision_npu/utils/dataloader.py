# Copyright (c) 2020 Huawei Technologies Co., Ltd
# All rights reserved.
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

import os
import threading
import itertools
import queue
import torch
import torch.distributed as dist
from torch.utils.data import _utils
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter as SrcSingleProcessDataLoaderIter
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter as SrcMultiProcessingDataLoaderIter
from torch.utils.data.dataloader import DataLoader as SrcDataLoader
from torch.utils.data.dataloader import _DatasetKind, _share_dist_seed, _get_distributed_settings
from torch.utils.data._utils.pin_memory import _pin_memory_loop
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
import torch.multiprocessing as multiprocessing
import torchvision

import torch_npu


MP_STATUS_CHECK_INTERVAL = 5.0


def npu_worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event,
                  auto_collation, collate_fn, drop_last, base_seed, init_fn, worker_id,
                  num_workers, persistent_workers, _shared_seed):
    # Only valid to data pre-processing processes in DVPP acceleration scenario
    # Set the current process without starting TBE tuning and compilation process to reduce host memory consumption
    os.environ["MIN_COMPILE_RESOURCE_USAGE_CTRL"] = "ub_fusion,coretype_check,op_compile"
    torch_npu.npu.set_device(torch_npu.npu.current_device())
    torchvision.set_image_backend('npu')
    torchvision.set_video_backend('npu')
    # Set priority: exlude AiCore, prefer DVPP
    torch_npu.npu.current_stream().set_data_preprocess_stream(True)
    _utils.worker._worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event,
                               auto_collation, collate_fn, drop_last, base_seed, init_fn, worker_id,
                               num_workers, persistent_workers, _shared_seed)


class DataLoader(SrcDataLoader):
    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)

    def __setattr__(self, attr, val):
        if self.__initialized and attr in (
                'batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset', 'persistent_workers'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(SrcDataLoader, self).__setattr__(attr, val)


class _SingleProcessDataLoaderIter(SrcSingleProcessDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        self._pin_memory = loader.pin_memory and torch_npu.npu.is_available()
        if self._timeout != 0:
            raise ValueError("self._timeout != 0")
        if self._num_workers != 0:
            raise ValueError("self._num_workers != 0")

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Taking care of distributed sharding
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            # For BC, use default SHARDING_PRIORITIES
            torch.utils.data.graph_settings.apply_sharding(self._dataset, self._world_size, self._rank)

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)


class _MultiProcessingDataLoaderIter(SrcMultiProcessingDataLoaderIter):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler"""

    def __init__(self, loader):
        try:
            torch_npu.npu.synchronize()
        except:
            pass
        self._prefetch_factor = loader.prefetch_factor
        self._dataset = loader.dataset
        self._shared_seed = None
        self._pg = None
        if isinstance(self._dataset, IterDataPipe):
            if dist.is_available() and dist.is_initialized():
                self._pg = dist.new_group(backend="gloo")
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(self._dataset, shared_rng)
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        ws, rank = _get_distributed_settings()
        self._world_size = ws
        self._rank = rank
        self._pin_memory = loader.pin_memory and torch.npu.is_available()
        self._pin_memory_device = loader.pin_memory_device
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self._profile_name = f"enumerate(DataLoader)#{self.__class__.__name__}.__next__"

        if self._num_workers <= 0:
            raise ValueError("self._num_workers <= 0")
        if self._prefetch_factor <= 0:
            raise ValueError("self._prefetch_factor <= 0")

        worker_loop = _utils.worker._worker_loop
        daemon = True
        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
            # if enable dvpp, worker process start method should be spawn and cannot be daemonic
            if torchvision.get_video_backend() == 'npu' or torchvision.get_image_backend() == 'npu':
                multiprocessing_context = multiprocessing.get_context('spawn')
                worker_loop = npu_worker_loop  # set device and priority
                daemon = False
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()
        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            index_queue = multiprocessing_context.Queue()  # type: ignore
            index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=worker_loop,
                args=(self._dataset_kind, self._dataset, index_queue,
                      self._worker_result_queue, self._workers_done_event,
                      self._auto_collation, self._collate_fn, self._drop_last,
                      self._base_seed, self._worker_init_fn, i, self._num_workers,
                      self._persistent_workers, self._shared_seed))
            w.daemon = daemon
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        if self._pin_memory:
            train_device_id = 0
            if torch_npu.npu.is_available():
                train_device_id = torch_npu.npu.current_device()
            else:
                train_device_id = torch.cuda.current_device()
            self._pin_memory_thread_done_event = threading.Event()

            self._data_queue = queue.Queue()  # type: ignore
            pin_memory_thread = threading.Thread(
                target=_pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      train_device_id,
                      self._pin_memory_thread_done_event, self._pin_memory_device))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        # In some rare cases, persistent workers (daemonic processes)
        # would be terminated before `__del__` of iterator is invoked
        # when main process exits
        # It would cause failure when pin_memory_thread tries to read
        # corrupted data from worker_result_queue
        # atexit is used to shutdown thread and child processes in the
        # right sequence before main process exits
        if self._persistent_workers and self._pin_memory:
            import atexit
            for w in self._workers:
                atexit.register(_MultiProcessingDataLoaderIter._clean_up_worker, w)

        # .pid can be None only before process is spawned (not the case, so ignore)
        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))  # type: ignore
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

    # staticmethod is used to remove reference to `_MultiProcessingDataLoaderIter`
    @staticmethod
    def _clean_up_worker(w):
        try:
            w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
        finally:
            if w.is_alive():
                w.terminate()

    def __del__(self):
        self._shutdown_workers()


def add_dataloader_method():
    torch.utils.data.DataLoader = DataLoader
    torch.utils.data.dataloader.DataLoader = DataLoader
    SrcDataLoader.__setattr__ = DataLoader.__setattr__
