import collections
import os
import random
import unittest
import cv2
import torch
import torch_npu
import torchvision
import torchvision.io as io
import torchvision_npu

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

        max_diff_threshold = 0
        abs_diff = torch.where(frames1 > frames2, frames1 - frames2, frames2 - frames1)
        if abs_diff.numel() == 0:
            return
        max_diff = abs_diff.max()
        self.assertLessEqual(max_diff, max_diff_threshold)

    def test_read_video_kunpeng_pts(self):
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            print("test_read_video_kunpeng_pts0 video path: ", full_path)

            # cpu
            torchvision.set_video_backend("pyav")
            os.environ["TORCHVISION_OMP_NUM_THREADS"] = "-1"
            video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path)
            # kunpeng
            os.environ["TORCHVISION_OMP_NUM_THREADS"] = "8"
            video_kunpeng, audio_kunpeng, info_kunpeng = torchvision.io.read_video(full_path)

            self._check_info(info_kunpeng, config)
            self._frames_compare(video_kunpeng, video_ori)
            self._frames_compare(audio_kunpeng, audio_ori)

    def test_read_video_kunpeng_random_pts_uint_sec(self):
        num_iter = 2
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            random_seek = [random.uniform(0, config.duration_sec) for _ in range(num_iter)]
            for _i in range(num_iter):
                end_pts = random.uniform(random_seek[_i], config.duration_sec)

                # cpu
                torchvision.set_video_backend("pyav")
                os.environ["TORCHVISION_OMP_NUM_THREADS"] = "-1"
                video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path, random_seek[_i], end_pts,
                                                                           pts_unit="sec")
                # kunpeng
                os.environ["TORCHVISION_OMP_NUM_THREADS"] = "8"
                video_kunpeng, audio_kunpeng, info_kunpeng = torchvision.io.read_video(full_path, random_seek[_i], end_pts, 
                                                                                       pts_unit="sec")
                self._frames_compare(video_kunpeng, video_ori)
                self._frames_compare(audio_kunpeng, audio_ori)

    def test_read_video_kunpeng_random_pts_uint_pts(self):
        num_iter = 2
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            random_seek = [random.randint(0, config.duration_pts) for _ in range(num_iter)]
            for _i in range(num_iter):
                end_pts = random.randint(random_seek[_i], config.duration_pts)
                #cpu
                os.environ["TORCHVISION_OMP_NUM_THREADS"] = "-1"
                video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path, random_seek[_i], end_pts,
                                                                           pts_unit="pts")
                # kunpeng
                os.environ["TORCHVISION_OMP_NUM_THREADS"] = "8"
                video_kunpeng, audio_kunpeng, info_kunpeng = torchvision.io.read_video(full_path, random_seek[_i], end_pts, 
                                                                                       pts_unit="pts")
                self._frames_compare(video_kunpeng, video_ori)
                self._frames_compare(audio_kunpeng, audio_ori)

    def test_read_video_kunpeng_equal_pts(self):
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            random_seek = random.randint(0, config.duration_pts)
            # cpu
            torchvision.set_video_backend("pyav")
            os.environ["TORCHVISION_OMP_NUM_THREADS"] = "-1"
            video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path, random_seek, random_seek)
            # kunpeng
            os.environ["TORCHVISION_OMP_NUM_THREADS"] = "8"
            video_kunpeng, audio_kunpeng, info_kunpeng = torchvision.io.read_video(full_path, random_seek, random_seek)
            self._frames_compare(video_kunpeng, video_ori)
            self._frames_compare(audio_kunpeng, audio_ori)

    def test_read_video_kunpeng_pts_out_of_range(self):
        test_video = "R6llTwEh07w.mp4"
        full_path = os.path.join(VIDEO_DIR, test_video)
        duration_pts = test_videos[test_video].duration_pts
        # kunpeng
        os.environ["TORCHVISION_OMP_NUM_THREADS"] = "8"
        # smaller than 0
        video_kunpeng, _, _ = torchvision.io.read_video(full_path, -15, -10, pts_unit="pts")
        self.assertEqual(len(video_kunpeng), 0)

        # out of video's max pts
        video_kunpeng, _, _ = torchvision.io.read_video(full_path, duration_pts + 10000, duration_pts + 15000,
                                                        pts_unit="pts")
        self.assertEqual(len(video_kunpeng), 1)

        # start larger than end
        self.assertRaises(ValueError, torchvision.io.read_video, full_path, 15, 10, pts_unit="pts")

    def test_read_video_kunpeng_chw(self):
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            # cpu
            torchvision.set_video_backend("pyav")
            os.environ["TORCHVISION_OMP_NUM_THREADS"] = "-1"
            video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path, output_format="TCHW")
            # kunpeng
            os.environ["TORCHVISION_OMP_NUM_THREADS"] = "8"
            video_kunpeng, audio_kunpeng, info_kunpeng = torchvision.io.read_video(full_path, output_format="TCHW")
            self._frames_compare(video_kunpeng, video_ori)

            # cpu
            os.environ["TORCHVISION_OMP_NUM_THREADS"] = "-1"
            video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path, -15, -10, pts_unit="pts",
                                                                       output_format="TCHW")
            # kunpeng
            os.environ["TORCHVISION_OMP_NUM_THREADS"] = "8"
            video_kunpeng, audio_kunpeng, info_kunpeng = torchvision.io.read_video(full_path, -15, -10, pts_unit="pts",
                                                                                   output_format="TCHW")
            self._frames_compare(video_kunpeng, video_ori)

    def test_read_video_kunpeng_hwc(self):
        for test_video, config in test_videos.items():
            full_path = os.path.join(VIDEO_DIR, test_video)
            # cpu
            torchvision.set_video_backend("pyav")
            os.environ["TORCHVISION_OMP_NUM_THREADS"] = "-1"
            video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path, output_format="THWC")
            # kunpeng
            os.environ["TORCHVISION_OMP_NUM_THREADS"] = "8"
            video_kunpeng, audio_kunpeng, info_kunpeng = torchvision.io.read_video(full_path, output_format="THWC")
            self._frames_compare(video_kunpeng, video_ori)

            # cpu
            os.environ["TORCHVISION_OMP_NUM_THREADS"] = "-1"
            video_ori, audio_ori, info_ori = torchvision.io.read_video(full_path, output_format="THWC")
            # kunpeng
            os.environ["TORCHVISION_OMP_NUM_THREADS"] = "8"
            video_kunpeng, audio_kunpeng, info_kunpeng = torchvision.io.read_video(full_path, output_format="THWC")
            self._frames_compare(video_kunpeng, video_ori)

    @unittest.skip("torchvision < 0.21.0 is incompatible with pyav >= 14.0.0")
    def test_invalid_file(self):
        # cpu
        torchvision.set_video_backend("pyav")
        os.environ["TORCHVISION_OMP_NUM_THREADS"] = "-1"
        self.assertRaises(RuntimeError, torchvision.io.read_video, "foo.mp4")
        # kunpeng
        os.environ["TORCHVISION_OMP_NUM_THREADS"] = "8"
        self.assertRaises(RuntimeError, torchvision.io.read_video, "foo.mp4")

if __name__ == "__main__":
    unittest.main()

