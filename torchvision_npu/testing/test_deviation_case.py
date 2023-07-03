import numpy as np
import torch
from torch_npu.testing.testcase import TestCase


class TestCase(TestCase):
    def assert_acceptable_deviation(self, a, b, deviation):
        if a.shape != b.shape:
            self.fail("shape error")
        if a.dtype != b.dtype:
            self.fail("dtype error")
        if a.dtype == torch.uint8:
            result = np.abs(a.to(torch.int16) - b.to(torch.int16))
        elif a.dtype == np.uint8:
            result = np.abs(a.astype(np.int16) - b.astype(np.int16))
        else:
            result = np.abs(a - b)
        if result.max() > deviation:
            self.fail(f"result error, got deviation {result.max()}")
