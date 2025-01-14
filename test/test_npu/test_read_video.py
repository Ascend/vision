import collections
import os
import random
import unittest
from unittest.mock import patch
import cv2
import torch
import torch_npu
import torchvision
import torchvision.io as io
import torchvision_npu

torch_npu.npu.current_stream().set_data_preprocess_stream(True)

try:
    import av
    io.video._check_av_available()
except ImportError:
    av = None

VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DataVideo", "readvideo")

CheckerConfig = [
    "duration_sec",
    "duration_pts",
    "video_fps",
    "audio_sample_rate",
]
GroundTruth = collections.namedtuple("GroundTruth", " ".join(CheckerConfig))

all_check_config = GroundTruth(
    duration_sec=0,
    duration_pts=0,
    video_fps=0,
    audio_sample_rate=0,
)

test_videos = {
    "R6llTwEh07w.mp4": GroundTruth(
        duration_sec=10.0,
        duration_pts=154624,
        video_fps=30.0,
        audio_sample_rate=44100,
    ),
    "WUzgd7C1pWA.mp4": GroundTruth(
        duration_sec=11.0,
        duration_pts=326326,
        video_fps=29.97,
        audio_sample_rate=48000,
    ),
}

no_support_videos = {
    "RATRACE_wave_f_nm_np1_fr_goo_37.avi": GroundTruth(
        duration_sec=2.0,
        duration_pts=0,
        video_fps=30.0,
        audio_sample_rate=None,
    ),
    "hmdb51_Turnk_r_Pippi_Michel_cartwheel_f_cm_np2_le_med_6.avi": None
}


class TestReadVideo(unittest.TestCase):
    def _check_info(self, info, config):
        self.assertNotEqual(info, None)
        self.assertAlmostEqual(info['video_fps'], config.video_fps, places=1)
        self.assertAlmostEqual(info['audio_fps'], config.audio_sample_rate, places=1)

    def _frames_compare(self, frames1, frames2):
        self.assertEqual(len(frames1), len(frames2))
        if len(frames1) == 0 or len(frames2) == 0:
            return
        self.assertEqual(frames1[0].shape, frames2[0].shape)

        max_diff_threshold = 3
        abs_diff = torch.where(frames1 > frames2, frames1 - frames2, frames2 - frames1)
        if abs_diff.numel() == 0:
            return
        max_diff = abs_diff.max()
        self.assertLessEqual(max_diff, max_diff_threshold)

    def test_read_video_npu_pts0(self):
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            print("test_read_video_npu_pts0 video path: ", full_path)

            # npu
            torchvision.set_video_backend('npu')
            video_npu, audio_npu, info_npu = torchvision.io.read_video(full_path)
            # cpu
            torchvision.set_video_backend('pyav')
            video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path)

            self._check_info(info_npu, config)
            video_npu = video_npu.cpu()
            audio_npu = audio_npu.cpu()
            self._frames_compare(video_npu, video_ori)
            self._frames_compare(audio_npu, audio_ori)

    def test_read_video_npu_random_pts_uint_sec(self):
        num_iter = 2
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            random_seek = [random.uniform(0, config.duration_sec) for _ in range(num_iter)]
            for _i in range(num_iter):
                end_pts = random.uniform(random_seek[_i], config.duration_sec)
                # npu
                torchvision.set_video_backend('npu')
                video_npu, audio_npu, info_npu = torchvision.io.read_video(full_path, random_seek[_i], end_pts,
                                                                           pts_unit="sec")
                # cpu
                torchvision.set_video_backend('pyav')
                video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path, random_seek[_i], end_pts,
                                                                           pts_unit="sec")
                video_npu = video_npu.cpu()
                audio_npu = audio_npu.cpu()
                self._frames_compare(video_npu, video_ori)
                self._frames_compare(audio_npu, audio_ori)

    def test_read_video_npu_random_pts_uint_pts(self):
        num_iter = 2
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            random_seek = [random.randint(0, config.duration_pts) for _ in range(num_iter)]
            for _i in range(num_iter):
                end_pts = random.randint(random_seek[_i], config.duration_pts)
                # npu
                torchvision.set_video_backend('npu')
                video_npu, audio_npu, info_npu = torchvision.io.read_video(full_path, random_seek[_i], end_pts,
                                                                           pts_unit="pts")
                # cpu
                torchvision.set_video_backend('pyav')
                video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path, random_seek[_i], end_pts,
                                                                           pts_unit="pts")
                video_npu = video_npu.cpu()
                audio_npu = audio_npu.cpu()
                self._frames_compare(video_npu, video_ori)
                self._frames_compare(audio_npu, audio_ori)

    def test_read_video_npu_equal_pts(self):
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            random_seek = random.randint(0, config.duration_pts)
            # npu
            torchvision.set_video_backend('npu')
            video_npu, audio_npu, info_npu = torchvision.io.read_video(full_path, random_seek, random_seek)
            # cpu
            torchvision.set_video_backend('pyav')
            video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path, random_seek, random_seek)
            video_npu = video_npu.cpu()
            audio_npu = audio_npu.cpu()
            self._frames_compare(video_npu, video_ori)
            self._frames_compare(audio_npu, audio_ori)

    def test_read_video_npu_pts_out_of_range(self):
        test_video = "R6llTwEh07w.mp4"
        full_path = os.path.join(VIDEO_DIR, test_video)
        duration_pts = test_videos[test_video].duration_pts
        # npu
        torchvision.set_video_backend('npu')
        # smaller than 0
        video_npu, _, _ = torchvision.io.read_video(full_path, -15, -10, pts_unit="pts")
        self.assertEqual(len(video_npu), 0)

        # out of video's max pts
        video_npu, _, _ = torchvision.io.read_video(full_path, duration_pts + 10000, duration_pts + 15000,
                                                    pts_unit="pts")
        self.assertEqual(len(video_npu), 0)

        # start lager than end
        self.assertRaises(ValueError, torchvision.io.read_video, full_path, 15, 10, pts_unit="pts")

    def test_read_video_npu_chw(self):
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            # npu
            torchvision.set_video_backend('npu')
            video_npu, audio_npu, info_npu = torchvision.io.read_video(full_path, output_format="TCHW")
            # cpu
            torchvision.set_video_backend('pyav')
            video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path, output_format="TCHW")
            video_npu = video_npu.cpu()
            self._frames_compare(video_npu, video_ori)

            # npu
            torchvision.set_video_backend('npu')
            video_npu, audio_npu, info_npu = torchvision.io.read_video(full_path, -15, -10, pts_unit="pts",
                                                                       output_format="TCHW")
            # cpu
            torchvision.set_video_backend('pyav')
            video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path, -15, -10, pts_unit="pts",
                                                                       output_format="TCHW")
            video_npu = video_npu.cpu()
            self._frames_compare(video_npu, video_ori)

    def test_read_video_npu_hwc(self):
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            # npu
            torchvision.set_video_backend('npu')
            video_npu, audio_npu, info_npu = torchvision.io.read_video(full_path, output_format="THWC")
            # cpu
            torchvision.set_video_backend('pyav')
            video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path, output_format="THWC")
            video_npu = video_npu.cpu()
            self._frames_compare(video_npu, video_ori)

            # npu
            torchvision.set_video_backend('npu')
            video_npu, audio_npu, info_npu = torchvision.io.read_video(full_path, -15, -10, pts_unit="pts",
                                                                       output_format="THWC")
            # cpu
            torchvision.set_video_backend('pyav')
            video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path, -15, -10, pts_unit="pts",
                                                                       output_format="THWC")
            video_npu = video_npu.cpu()
            self._frames_compare(video_npu, video_ori)

    def test_read_video_npu_not_support_type(self):
        for test_video, config in no_support_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            torchvision.set_video_backend('npu')
            video_npu, audio_npu, info_npu = torchvision.io.read_video(full_path)
            torchvision.set_video_backend('pyav')
            video_cpu, audio_cpu, info_cpu = torchvision.io.read_video(full_path)
            self.assertTrue(torch.equal(video_npu.cpu(), video_cpu))
            self.assertTrue(torch.equal(audio_npu.cpu(), audio_cpu))
            self.assertEqual(info_cpu, info_npu)

    def test_read_video_mem(self):
        torchvision.set_video_backend('npu')
        full_path = os.path.join(VIDEO_DIR, list(test_videos.keys())[0])
        cap = cv2.VideoCapture(full_path)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        cap.release()
        vframes, _, _ = torchvision.io.read_video(full_path)
        max_mem = torch_npu.npu.max_memory_allocated()
        cal_mem = 3 * frame_height * frame_width * frame_count
        # reserve 10%
        reserve_mem_ratio = 0.1
        self.assertLess(max_mem, (1 + reserve_mem_ratio) * cal_mem)


if __name__ == '__main__':
    unittest.main()
