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
import numpy as np
import torch
import torch_npu
from torchvision.utils import _log_api_usage_once
import torchvision
import av
from torchvision import io as IO
import torchvision_npu
from torchvision_npu._utils import PathManager
from torchvision_npu import io as IO_npu


try:
    FFmpegError = av.FFmpegError
except AttributeError:
    FFmpegError = av.AVError


def patch_io_video():
    setattr(torchvision.io, "read_video_ori", torchvision.io.read_video)
    torchvision.io.read_video = read_video


def is_neon_supported():
    cpu_info = "/proc/cpuinfo"
    neon_support = False
    if os.path.exists(cpu_info):
        with open(cpu_info, "r") as f:
            for line in f:
                if "neon" in line.lower() or "asimd" in line.lower():
                    neon_support = True
                    break
    return neon_support


def get_kp_thread_num():
    num_thread = -1
    env_var = os.environ.get("TORCHVISION_OMP_NUM_THREADS")
    if env_var is not None and env_var.isdigit() and is_neon_supported():
        num_thread = int(env_var)
    return num_thread


def read_video(
        filename: str,
        start_pts: Union[float, Fraction] = 0,
        end_pts: Optional[Union[float, Fraction]] = None,
        pts_unit: str = "pts",
        output_format: str = "THWC",
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
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """
    num_thread = get_kp_thread_num()
    if torchvision.get_video_backend() == 'npu':
        return IO_npu._video._read_video(filename, start_pts, end_pts, pts_unit, output_format)
    elif torchvision.get_video_backend() == 'pyav' and num_thread > 0:
        return IO_npu._video.kp_read_video(filename, start_pts, end_pts, pts_unit, output_format, thread_count=num_thread)
    return IO.read_video_ori(filename, start_pts, end_pts, pts_unit, output_format)


class VideoFrameDvpp:
    dts: int
    pts: int
    positions: int
    frame: np.ndarray

    def __init__(
            self, dts: int = 0, pts: int = 0
    ) -> None:
        self.pts = pts
        self.dts = dts
        ...


class DecodeParams:
    def __init__(self, container, start_offset, end_offset, stream):
        self.container = container
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.stream = stream


def get_frame_by_cv(filename: str, container: "av.container.Container", stream: "av.stream.Stream") -> Tuple[Dict[
    int, VideoFrameDvpp], int]:
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_FORMAT, -1)
    frames: Dict[int, VideoFrameDvpp] = {}
    pts_list = []
    pts_per_frame = round(1 / cap.get(cv2.CAP_PROP_POS_AVI_RATIO) / cap.get(cv2.CAP_PROP_FPS), 0)

    for packet in container.demux(stream):
        if packet.pts is not None:
            frame = VideoFrameDvpp(packet.dts, packet.pts)
            ret, frame.frame = cap.read()
            pts_list.append(packet.pts)
            frames[frame.pts] = frame
    cap.release()
    pts_list.sort()
    position_list = {value: index for index, value in enumerate(pts_list)}
    for frame in frames.values():
        frame.positions = position_list[frame.pts]

    return frames, pts_per_frame


def _decode_video_dvpp(
        decode_params: DecodeParams,
        frames: Dict[int, VideoFrameDvpp],
        pts_per_frame: int
) -> torch.Tensor:
    container = decode_params.container
    start_offset = decode_params.start_offset
    end_offset = decode_params.end_offset
    stream = decode_params.stream

    codecs_type = stream.name
    hi_pt_h264 = 96
    hi_pt_h265 = 265
    if codecs_type == "h264":
        codec_id = hi_pt_h264
    elif codecs_type == "hevc":
        codec_id = hi_pt_h265
    else:
        raise ValueError(f"The video codecs_type should be either 'h264' or 'hevc', got {codecs_type}.")

    # if start_offset is between 2 frames, get one more previous frame
    start_offset = int(start_offset / pts_per_frame) * pts_per_frame

    frame_width = stream.width
    frame_height = stream.height

    # update end_offset_real
    end_offset_real = end_offset
    # if end_offset equals to a frame's pts, get one more this frame
    if end_offset_real % pts_per_frame == 0:
        end_offset_real += 1
    end_offset_real = min(end_offset_real, len(frames) * pts_per_frame)
    start_frame = math.ceil(start_offset / pts_per_frame)
    total_frame = math.ceil((end_offset_real - start_offset) / pts_per_frame)

    if end_offset_real < start_offset or total_frame == 0:
        return torch.empty(0, dtype=torch.uint8, device='npu')
    ret_tensor = torch.empty([total_frame, 3, frame_height, frame_width], dtype=torch.uint8, device='npu')

    # decode from dvpp
    chn = torch.ops.torchvision._decode_video_create_chn(codec_id)
    if chn == -1:
        warnings.warn(f"_decode_video_create_chn failed {chn}")
        torch.ops.torchvision._dvpp_sys_exit()
        return None
    ret = torch.ops.torchvision._decode_video_start_get_frame(chn, total_frame)
    if not ret == 0:
        warnings.warn(f"_decode_video_start_get_frame failed {ret}")
        torch.ops.torchvision._decode_video_destroy_chnl(chn)
        torch.ops.torchvision._dvpp_sys_exit()
        return None

    for packet in container.demux(stream):
        if packet.pts is not None:
            frame = frames[packet.pts].frame
            input_tensor = torch.tensor(frame).npu(non_blocking=True)

            if start_offset <= int(packet.pts) <= end_offset:
                display = True
                output_tensor = ret_tensor[frames[packet.pts].positions - start_frame]
            else:
                display = False
                output_tensor = torch.empty(0, dtype=torch.uint8, device='npu')
            # 12:rgb888packed; 13:bgr888packed; 69:rgb888planer; 70:bgr888planer. Packed is HWC, planer is CHW
            # use CHW to avoid memory copy
            ret = torch.ops.torchvision._decode_video_send_stream(chn, input_tensor, 69, display,
                                                                  output_tensor)
            if not ret == 0:
                warnings.warn(f"_decode_video_send_stream failed {ret}")

    # ret_tensor is ordered by pts
    ret_tensor_dvpp = torch.ops.torchvision._decode_video_stop_get_frame(chn, total_frame)

    # if ret_tensor_dvpp empty, means ret_tensor already filled
    if ret_tensor_dvpp.numel() != 0:
        ret_tensor = ret_tensor_dvpp

    ret = torch.ops.torchvision._decode_video_destroy_chnl(chn)
    if not ret == 0:
        warnings.warn(f"_decode_video_destroy_chnl failed {ret}")

    return ret_tensor


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

    max_buffer_size = 5
    should_buffer = True
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
            should_buffer = obj.group(3) == b"p"

    seek_offset = start_offset
    # some files don't seek to the right location, so better be safe here
    seek_offset = max(seek_offset - 1, 0)

    if should_buffer:
        seek_offset = max(seek_offset - max_buffer_size, 0)
    # init frames before seek
    frames, pts_per_frame = get_frame_by_cv(filename, container, stream)
    try:
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except FFmpegError:
        return []
    decode_params = DecodeParams(container, start_offset, end_offset, stream)
    frames = _decode_video_dvpp(decode_params, frames, pts_per_frame)

    if frames is None:
        warnings.warn(f"_decode_video_dvpp failed: {filename}")

    return frames


def _read_video(
        filename: str,
        start_pts: Union[float, Fraction] = 0,
        end_pts: Optional[Union[float, Fraction]] = None,
        pts_unit: str = "pts",
        output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(_read_video)

    if pts_unit not in ("pts", "sec"):
        raise ValueError(f"pts_unit should be either 'pts' or 'sec', got {pts_unit}.")

    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    filepath = os.path.realpath(filename)
    PathManager.check_directory_path_readable(filepath)

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
        with av.open(filepath, metadata_errors="ignore") as container:
            if container.streams.audio:
                audio_timebase = container.streams.audio[0].time_base

            if container.streams.video:
                if container.streams.video[0].name not in ("hevc", "h264"):
                    warnings.warn(f"This video in {filepath} is coding by {container.streams.video[0].name}, "
                                   "not supported on DVPP backend and will fall back to run on the pyav."
                                   "This may have performance implications.")
                    torchvision.set_video_backend('pyav')
                    vframes, aframes, info = IO.video.read_video(
                        filepath,
                        start_pts,
                        end_pts,
                        pts_unit,
                        output_format,
                    )
                    torchvision.set_video_backend('npu')
                    return vframes.npu(), aframes.npu(), info
                else:
                    vframes = _read_from_stream_dvpp(
                        filepath,
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
            else:
                vframes = torch.empty(0, dtype=torch.uint8, device='npu')

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

    except FFmpegError:
        vframes = torch.empty(0, dtype=torch.uint8, device='npu')

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

    if output_format == "THWC" and vframes is not None and len(vframes) != 0:
        # [T,C,H,W] --> [T,H,W,C]
        vframes = vframes.permute(0, 2, 3, 1)

    return vframes, aframes, info


def _kp_yuv2rgb_is_vaild(video_pix_fmt, frame, video_colorspace):
    return video_pix_fmt in ["yuv420p", "yuvj420p"] and \
           not frame.width % 2 and not frame.height % 2 and \
           video_colorspace in [1, 2]


def kp_read_video(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
    thread_count: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(read_video)

    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

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
    video_frames = []
    audio_frames = []
    audio_timebase = torchvision.io._video_opt.default_timebase

    try:
        with av.open(filename, metadata_errors="ignore") as container:
            if container.streams.audio:
                audio_timebase = container.streams.audio[0].time_base
            if container.streams.video:
                container.streams.video[0].thread_type = "AUTO"
                container.streams.video[0].thread_count = thread_count
                video_frames = torchvision.io.video._read_from_stream(
                    container,
                    start_pts,
                    end_pts,
                    pts_unit,
                    container.streams.video[0],
                    {"video": 0},
                )
                video_fps = container.streams.video[0].average_rate
                video_pix_fmt = container.streams.video[0].pix_fmt
                video_color_range = container.streams.video[0].color_range
                video_colorspace = container.streams.video[0].colorspace
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

    except FFmpegError as e:
        warnings.warn(f"[Warning] Error while reading video {filename}: {e}")

    vframes_list = []
    src_stride = [0, 0, 0, 0]
    for index, frame in enumerate(video_frames):
        y_plane = torch.from_numpy(np.frombuffer(frame.planes[0], dtype=np.uint8))
        u_plane = torch.from_numpy(np.frombuffer(frame.planes[1], dtype=np.uint8))
        v_plane = torch.from_numpy(np.frombuffer(frame.planes[2], dtype=np.uint8))
        for i in range(3):
            src_stride[i] = frame.planes[i].line_size
        if _kp_yuv2rgb_is_vaild(video_pix_fmt, frame, video_colorspace):
            vframes_list.append(torch.ops.torchvision.yuv2rgb(thread_count,
                                                              frame.width,
                                                              frame.height,
                                                              video_color_range,
                                                              video_colorspace,
                                                              y_plane,
                                                              src_stride,
                                                              u_plane,
                                                              v_plane).numpy())
        else:
            vframes_list.append(video_frames[index].to_rgb().to_ndarray())

    aframes_list = [frame.to_ndarray() for frame in audio_frames]

    if vframes_list:
        vframes = torch.as_tensor(np.stack(vframes_list))
    else:
        vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    if aframes_list:
        aframes = np.concatenate(aframes_list, 1)
        aframes = torch.as_tensor(aframes)
        if pts_unit == "sec":
            start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
            if end_pts != float("inf"):
                end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
        aframes = torchvision.io.video._align_audio_frames(aframes, audio_frames, start_pts, end_pts)
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info
