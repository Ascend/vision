import os
import contextlib
import unittest
import argparse
import sys
import random

from PIL import Image
import __main__
import numpy as np
import torch

IS_PY39 = sys.version_info.major == 3 and sys.version_info.minor == 9
PY39_SEGFAULT_SKIP_MSG = "Segmentation fault with Python 3.9, see the 3367 issue of pytorch vision."
PY39_SKIP = unittest.skipIf(IS_PY39, PY39_SEGFAULT_SKIP_MSG)


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


ACCEPT = os.getenv('EXPECTTEST_ACCEPT')
TEST_WITH_SLOW = os.getenv('PYTORCH_TEST_WITH_SLOW', '0') == '1'


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--accept', action='store_true')
args, remaining = parser.parse_known_args()
if not ACCEPT:
    ACCEPT = args.accept
for i, arg in enumerate(sys.argv):
    if arg == '--accept':
        del sys.argv[i]
        break


class MapNestedTensorObjectImpl(object):
    def __init__(self, tensor_map_fn):
        self.tensor_map_fn = tensor_map_fn

    def __call__(self, obj):
        if isinstance(obj, torch.Tensor):
            return self.tensor_map_fn(obj)

        elif isinstance(obj, dict):
            mapped_dict = {}
            for key, value in obj.items():
                mapped_dict[self(key)] = self(value)
            return mapped_dict

        elif isinstance(obj, (list, tuple)):
            mapped_iter = []
            for item in obj:
                mapped_iter.append(self(item))
            return mapped_iter if not isinstance(obj, tuple) else tuple(mapped_iter)

        else:
            return obj


def map_nested_tensor_object(obj, tensor_map_fn):
    impl = MapNestedTensorObjectImpl(tensor_map_fn)
    return impl(obj)


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


@contextlib.contextmanager
def freeze_rng_state():
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    yield
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)
    torch.set_rng_state(rng_state)


class TransformsTester(unittest.TestCase):

    def _create_data(self, height=3, width=3, channels=3, device="cpu"):
        tensor = torch.randint(0, 256, (channels, height, width), dtype=torch.uint8, device=device)
        pil_img = Image.fromarray(tensor.permute(1, 2, 0).contiguous().cpu().numpy())
        return tensor, pil_img

    def _create_data_batch(self, height=3, width=3, channels=3, num_samples=4, device="cpu"):
        batch_tensor = torch.randint(
            0, 256,
            (num_samples, channels, height, width),
            dtype=torch.uint8,
            device=device
        )
        return batch_tensor

    def compareTensorToPIL(self, tensor, pil_image, msg=None):
        np_pil_image = np.array(pil_image)
        if np_pil_image.ndim == 2:
            np_pil_image = np_pil_image[:, :, None]
        pil_tensor = torch.as_tensor(np_pil_image.transpose((2, 0, 1)))
        if msg is None:
            msg = "tensor:\n{} \ndid not equal PIL tensor:\n{}".format(tensor, pil_tensor)
        self.assertTrue(tensor.cpu().equal(pil_tensor), msg)

    def approxEqualTensorToPIL(self, tensor, pil_image, tol=1e-5, msg=None, agg_method="mean"):
        np_pil_image = np.array(pil_image)
        if np_pil_image.ndim == 2:
            np_pil_image = np_pil_image[:, :, None]
        pil_tensor = torch.as_tensor(np_pil_image.transpose((2, 0, 1))).to(tensor)
        # error value can be mean absolute error, max abs error
        err = getattr(torch, agg_method)(torch.abs(tensor - pil_tensor)).item()
        self.assertLess(
            err, tol,
            msg="{}: err={}, tol={}: \n{}\nvs\n{}".format(msg, err, tol, tensor[0, :10, :10], pil_tensor[0, :10, :10])
        )


def cycle_over(objs):
    for idx, obj in enumerate(objs):
        yield obj, objs[:idx] + objs[idx + 1:]


def int_dtypes():
    return torch.testing.integral_types()


def float_dtypes():
    return torch.testing.floating_types()
