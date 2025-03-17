import os
import random
import unittest
import numpy as np
import torch
import torch_npu
import torchvision
from torchvision.transforms import v2
import torchvision_npu


class TestUniformTemporalSubsample(unittest.TestCase):
    def _reference_uniform_temporal_subsample_video(self, video, *, num_samples):
        t = video.shape[-4]
        assert num_samples > 0 and t > 0
        indices = torch.linspace(0, t - 1, num_samples, device=video.device, dtype=torch.float64)
        indices = torch.clamp(indices, 0, t - 1).long()
        return torch.index_select(video, -4, indices)
    
    def generate_random_video_tensor(self, num_frames, num_channels, height, width):
        video_float = np.random.rand(num_frames, num_channels, height, width)
        video_int = np.random.randint(0, 255, size=(num_frames, num_channels, height, width), dtype=np.uint8)
        video_tensor_float = torch.from_numpy(video_float)
        video_tensor_int = torch.from_numpy(video_int)
        return video_tensor_float, video_tensor_int
    
    def test_video_correctness(self):
        os.environ["TORCHVISION_OMP_NUM_THREADS"] = "8"
        test_nums = 10
        for i in range(test_nums):
            num_channels = 3
            num_frames = np.random.randint(2, 50)
            height = np.random.randint(480, 1024)
            width = np.random.randint(480, 1024)
            num_samples = np.random.randint(1, num_frames)
            video_tensor_float, video_tensor_int = self.generate_random_video_tensor(num_frames, num_channels, height, width)
            actual_float = v2.functional.uniform_temporal_subsample_video(video_tensor_float, num_samples=num_samples)
            expected_float = self._reference_uniform_temporal_subsample_video(video_tensor_float, num_samples=num_samples)
            self.assertTrue(torch.allclose(actual_float, expected_float, atol=0))
            actual_int = v2.functional.uniform_temporal_subsample_video(video_tensor_int, num_samples=num_samples)
            expected_int = self._reference_uniform_temporal_subsample_video(video_tensor_int, num_samples=num_samples)
            self.assertTrue(torch.allclose(actual_int, expected_int, atol=0))

if __name__ == "__main__":
    unittest.main()
