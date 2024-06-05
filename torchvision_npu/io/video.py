# Copyright (c) 2024, Huawei Technologies.All rights reserved.
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

import math
import warnings
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from fractions import Fraction
import os

import cv2
import av
import numpy as np
import torch
import torch_npu
from torchvision.utils import _log_api_usage_once
import torchvision
from torchvision import io as IO
import torchvision_npu
from torchvision_npu import io as IO_npu


def patch_io_video():
    setattr(torchvision.io, "read_video_ori", torchvision.io.read_video)
    torchvision.io.read_video = read_video


def read_video(
        filename: str,
        start_pts: Union[float, Fraction] = 0,
        end_pts: Optional[Union[float, Fraction]] = None,
        pts_unit: str = "pts",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """

    if torchvision.get_video_backend() == 'npu':
        return IO_npu.video._read_video(filename, start_pts, end_pts, pts_unit)
    return IO.read_video_ori(filename, start_pts, end_pts, pts_unit)


class VideoFrameDvpp:
    dts: int
    pts: int

    def __init__(
            self, dts: int = 0, pts: int = 0
    ) -> None:
        self.pts = pts
        self.dts = dts
        ...


_should_buffer = True


def get_frame_by_cv(filename: str, container: "av.container.Container", stream: "av.stream.Stream") -> Dict[
    int, VideoFrameDvpp]:
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_FORMAT, -1)
    frames = {}
    for packet in container.demux(stream):
        if packet.pts is not None:
            frame = VideoFrameDvpp(packet.dts, packet.pts)
            ret, frame.frame = cap.read()
            frames[frame.pts] = frame
    cap.release()
    return frames


def _decode_video_dvpp(
        container: "av.container.Container",
        start_offset: float,
        end_offset: float,
        stream: "av.stream.Stream",
        frames: Dict[int, VideoFrameDvpp]
) -> torch.Tensor:
    codecs_type = stream.name
    hi_pt_h264 = 96
    hi_pt_h265 = 265
    if codecs_type == "h264":
        codec_id = hi_pt_h264
    elif codecs_type == "hevc":
        codec_id = hi_pt_h265
    else:
        raise ValueError(f"The video codecs_type should be either 'h264' or 'hevc', got {codecs_type}.")

    # decode from dvpp
    chn = torch.ops.torchvision._decode_video_create_chn(codec_id)
    if chn == -1:
        torch.ops.torchvision._dvpp_sys_exit()
        return None
    ret = torch.ops.torchvision._decode_video_start_get_frame(chn)
    if not ret == 0:
        warnings.warn(f"_decode_video_start_get_frame failed {ret}")
        torch.ops.torchvision._decode_video_destroy_chnl(chn)
        torch.ops.torchvision._dvpp_sys_exit()
        return None

    # if start_offset is between 2 frames, get one more previous frame
    pts_per_frame = 0
    if stream.time_base * stream.average_rate != 0:
        pts_per_frame = int(1 / (stream.time_base * stream.average_rate))
    start_offset -= pts_per_frame

    max_buffer_size = 5
    frame_width = stream.width
    frame_height = stream.height
    for packet in container.demux(stream):
        if packet.pts is not None:
            # pts may smaller than dts, so buffer some frames
            if _should_buffer and packet.pts > end_offset + max_buffer_size * pts_per_frame:
                break
            elif not _should_buffer and packet.pts > end_offset:
                break
            frame = frames[packet.pts].frame
            input_tensor = torch.tensor(frame).npu(non_blocking=True)
            output_tensor = torch.empty([1, 3, frame_height, frame_width], dtype=torch.uint8).npu(
                non_blocking=True)
            if start_offset < int(packet.pts) <= end_offset:
                display = True
            else:
                display = False
            # 12:rgb888packed; 13:bgr888packed; 69:rgb888planer; 70:bgr888planer. Packed is HWC, planer is CHW
            # use CHW to avoid memory copy
            ret = torch.ops.torchvision._decode_video_send_stream(chn, input_tensor, 69, display,
                                                                  output_tensor)
            if not ret == 0:
                warnings.warn(f"_decode_video_send_stream failed {ret}")

    # ret_tensors_list is ordered by pts
    ret_tensors = torch.ops.torchvision._decode_video_stop_get_frame(chn)

    ret = torch.ops.torchvision._decode_video_destroy_chnl(chn)
    if not ret == 0:
        warnings.warn(f"_decode_video_destroy_chnl failed {ret}")

    return ret_tensors


def _read_from_stream_dvpp(
        filename: str,
        container: "av.container.Container",
        start_offset: float,
        end_offset: float,
        pts_unit: str,
        stream: "av.stream.Stream",
        stream_name: Dict[str, Optional[Union[int, Tuple[int, ...], List[int]]]],
) -> torch.Tensor:
    if not stream.type == "video":
        raise RuntimeError("_read_from_stream_dvpp only handle video type")
    if pts_unit == "sec" and stream.time_base != 0:
        start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
        if end_offset != float("inf") and stream.time_base != 0:
            end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))
    else:
        warnings.warn("The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.")

    global _should_buffer
    max_buffer_size = 5

    # DivX-style packed B-frames can have out-of-order pts (2 frames in a single pkt)
    # so need to buffer some extra frames to sort everything
    # properly
    extradata = stream.codec_context.extradata
    # overly complicated way of finding if `divx_packed` is set, following
    if extradata and b"DivX" in extradata:
        # can't use regex directly because of some weird characters sometimes...
        pos = extradata.find(b"DivX")
        data = extradata[pos:]
        obj = re.search(rb"DivX(\d+)Build(\d+)(\w)", data)
        if obj is None:
            obj = re.search(rb"DivX(\d+)b(\d+)(\w)", data)
        if obj is not None:
            _should_buffer = obj.group(3) == b"p"

    seek_offset = start_offset
    # some files don't seek to the right location, so better be safe here
    seek_offset = max(seek_offset - 1, 0)

    if _should_buffer:
        seek_offset = max(seek_offset - max_buffer_size, 0)
    # init frames before seek
    frames = get_frame_by_cv(filename, container, stream)
    try:
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except av.AVError:
        return []

    frames = _decode_video_dvpp(container, start_offset, end_offset, stream, frames)

    if frames is None:
        warnings.warn(f"_decode_video_dvpp failed: {filename}")

    return frames


def _read_video(
        filename: str,
        start_pts: Union[float, Fraction] = 0,
        end_pts: Optional[Union[float, Fraction]] = None,
        pts_unit: str = "pts",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(_read_video)

    if pts_unit not in ("pts", "sec"):
        raise ValueError(f"pts_unit should be either 'pts' or 'sec', got {pts_unit}.")

    if not os.path.exists(filename):
        raise RuntimeError(f"File not found: {filename}")

    torchvision.io.video._check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError(
            f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}"
        )

    info = {}
    audio_frames = []
    audio_timebase = torchvision.io._video_opt.default_timebase

    try:
        with av.open(filename, metadata_errors="ignore") as container:
            if container.streams.audio:
                audio_timebase = container.streams.audio[0].time_base
            time_base = torchvision.io._video_opt.default_timebase
            if container.streams.video:
                time_base = container.streams.video[0].time_base
            elif container.streams.audio:
                time_base = container.streams.audio[0].time_base
            # video_timebase is the default time_base
            start_pts, end_pts, pts_unit = torchvision.io._video_opt._convert_to_sec(start_pts, end_pts, pts_unit, time_base)
            if container.streams.video:
                if container.streams.video[0].name not in ("hevc", "h264"):
                    raise RuntimeError(f"This video is coding by {container.streams.video[0].name}, not supported. "
                                       f"Only support: h264, hevc.")

                vframes = _read_from_stream_dvpp(
                    filename,
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.video[0],
                    {"video": 0},
                )
                video_fps = container.streams.video[0].average_rate
                # guard against potentially corrupted files
                if video_fps is not None:
                    info["video_fps"] = float(video_fps)

            if container.streams.audio:
                audio_frames = torchvision.io.video._read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.audio[0],
                    {"audio": 0},
                )
                info["audio_fps"] = container.streams.audio[0].rate

    except av.AVError:
        raise RuntimeWarning("Catch an AVError") from av.AVError

    aframes_list = [frame.to_ndarray() for frame in audio_frames]

    if aframes_list:
        aframes = np.concatenate(aframes_list, 1)
        aframes = torch.as_tensor(aframes)
        if pts_unit == "sec" and audio_timebase != 0:
            start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
            if end_pts != float("inf") and audio_timebase != 0:
                end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
        aframes = torchvision.io.video._align_audio_frames(aframes, audio_frames, start_pts, end_pts)
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    aframes = aframes.npu(non_blocking=True)

    if vframes is not None and len(vframes) != 0:
        # default format is THWC [T,C,H,W] --> [T,H,W,C]
        vframes = vframes.permute(0, 2, 3, 1)

    return vframes, aframes, info
