from __future__ import annotations

"""Unreal Go2 episode -> LeRobot v2.1 转换入口。

典型用法：
    .\\.venv-py311\\Scripts\\python.exe unreal.py ^
        --raw_dir F:/UnrealProject/Saved/scene_0004 ^
        --output_dir ./tmp/saved_scene_0004_direct_output ^
        --camera_keys front,rear,left,right ^
        --num_processes 1 ^
        --skip_invalid_episodes ^
        --trim_extra_tail_frame ^
    --resume ^
    --skip_depth ^
    --copy_rgb_mp4

`--raw_dir` 可传三种层级：
    1. UE OutputRoot，例如 C:/Data/Saved
    2. 某个 scene/user 目录，例如 C:/Data/Saved/scene_0002/szt
    3. 单个 episode 目录，例如 C:/Data/Saved/scene_0002/szt/episode_000000

脚本会递归查找 `episode_meta.json`，只转换 `status == "completed"` 且存在
`frames.jsonl` 的 episode。默认导出 front/rear/left/right 四路 RGB；可用
`--camera_keys front` 或 `--camera_keys front,left,right` 选择子集。
如果输入根目录混有旧格式或不完整 episode，可加 `--skip_invalid_episodes`
跳过不兼容条目。转换报告会写入 `meta/unreal_conversion_report.json`，其中记录
已准备提交、实际成功落盘、失败或跳过的 episode。
`--output_dir` 直接作为总输出目录；默认按 scene 分组，每个 scene 目录内
包含一份标准 LeRobot 数据集和额外 sidecar：
    output/
        scene_0001/
            data/
            meta/
            videos/
            episodes_extras.parquet
            images/
        scene_0002/
            ...

如果同一个 raw_dir 中混有多套 fps/分辨率，可加 `--split_by_schema`，脚本会按
`sample_rate_hz + capture_height + capture_width` 再拆一层 schema 目录，例如：
    output/
        fps10_480x640/
            scene_0001/
        fps30_720x1280/
            scene_0001/
转换总报告写入：
    output/unreal_conversion_report.json

如果旧数据存在“frames.jsonl 比 episode_meta.frame_count 多 1 行”的半帧问题，可加
`--trim_extra_tail_frame`。脚本只会在最后一行 frame_index 正好等于 frame_count 时，
在内存中裁掉最后一行，不会修改原始 episode 文件。

如果长任务中断，可加 `--resume` 跳过已完成的 schema/scene 输出。遇到半成品 scene
时默认报错；确认可重建该 scene 时再加 `--overwrite_incomplete` 移走半成品并重跑。
`--resume` 会在扫描结束后写出 `output_dir/unreal_scan_cache.pkl`；之后可加
`--reuse_scan_cache` 直接复用扫描结果，跳过逐 episode 外参校验。
如果 RGB 源数据本身是帧数匹配的 MP4，可加 `--copy_rgb_mp4` 直接复制到 LeRobot
视频目录，避免解码和重新编码；需要裁掉额外尾帧的 episode 会自动回退到重编码。

输入 episode 需要包含：
    episode_meta.json
    frames.jsonl
    rgb/<camera>.mp4 或 rgb/<camera>/<00000>.png 序列
    task_info.csv（可选）

输出 LeRobot 数据集包含：
    meta/info.json
    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/episodes_extras.jsonl
    data/chunk-000/episode_*.parquet
    videos/chunk-000/video.<camera>/episode_*.mp4
scene 目录下还会额外写出：
    episodes_extras.parquet  # 每条 episode 一行，含 K_<camera>、Extrinsic_<camera> 等
    images/chunk-000/observation.depth.<camera>/episode_*/00000.png
        # 默认从 depth/<camera>.mp4 的 HueMp4 编码恢复为 uint16 毫米深度；
        # 加 `--skip_depth` 时不读取、不校验、不导出 depth sidecar。

每帧 parquet 字段：
    annotation.human.action.task_description
    observation.state  # [tx, ty, tz, qx, qy, qz, qw], 单位 m, 四元数 xyzw
    action             # 当前复制 observation.state

坐标约定：
    UE 输入：位置 cm，机体系/相机系均为 +X 前、+Y 右、+Z 上。
    输出：位置 m，机体系 +X 前、+Y 左、+Z 上；相机系为 OpenCV +X 右、+Y 下、+Z 前。
    trajectory 的 world 坐标系固定为第一帧机体坐标系，因此第一帧 state 应接近
    [0, 0, 0, 0, 0, 0, 1]。

外参处理：
    `video.<camera>.body_from_camera` 是 episode 级 metadata。因为 UE 每帧都写
    `pose` 和 `camera_pose_<camera>`，本脚本会逐帧反算 T_body<-camera，并严格检查
    一个 episode 内外参是否固定。容差由 `--extrinsic_tolerance_translation_m` 和
    `--extrinsic_tolerance_rotation_deg` 控制。
"""

import argparse
import csv
import json
import logging
import math
import os
import pickle
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation

# 限制 OpenCV 解码线程：OpenCV 默认按机器全核开线程池/ffmpeg 解码线程，多 worker 或
# 多实例并行时会造成线程超额订阅、调度颠簸。这里统一限制为小值（可用 XNAV_DECODE_THREADS
# 覆盖）。单 worker 本就只解码一个 episode，并行度由 worker 数提供，无需每个解码占满核。
RGB_DECODE_THREADS = max(1, int(os.environ.get("XNAV_DECODE_THREADS", "2")))
# ffmpeg 解码线程数必须在 VideoCapture 创建前通过该环境变量设置（fork 后被子进程继承）。
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"threads;{RGB_DECODE_THREADS}")

TASK_DESCRIPTION_KEY = "annotation.human.action.task_description"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
POSE_AXES = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
DEFAULT_CAMERA_KEYS = ("front", "rear", "left", "right")
DEPTH_DARK_THRESHOLD = 16
DEPTH_SATURATION_THRESHOLD = 16
SCAN_CACHE_VERSION = 1

CLEANING_MIN_SPEED_CM_SEC = 20.0
CLEANING_DEVIATION_ANGLE_DEG = 30.0
CLEANING_TIME_CONSTANT = 0.7
CLEANING_RECOVERY_TIME_CONSTANT = 0.15
CLEANING_ENTER_SOFT = 0.55
CLEANING_ENTER_HARD = 0.85
CLEANING_EXIT_HARD = 0.40
CLEANING_EXIT_SOFT = 0.20
CLEANING_TELEPORT_RESET_CM = 500.0
CLEANING_MAX_VIOLATION_RATIO = 0.05
CLEANING_HARD_MIN_DURATION_SEC = 1.0
CLEANING_MAX_HARD_RATIO = 0.02
CLEANING_MAX_STUCK_RATIO = 0.10
CLEANING_STUCK_WINDOW_SEC = 5.0
CLEANING_STUCK_MAX_DISPLACEMENT_CM = 50.0
# 每个静止段保留的代表帧数量（与时长无关）。固定计数而非按帧率降采样，
# 避免极长静止段降采样后仍残留大量静止帧。
CLEANING_STUCK_KEEP_FRAMES_PER_SEGMENT = 2
CLEANING_STUCK_BOUNDARY_KEEP_FRAMES = 1
CLEANING_STUCK_YAW_PRESERVE_THRESHOLD_DEG = 10.0
CLEANING_MIN_CLEANED_FRAMES = 2
# 整集级退化静止判据：整段轨迹 XY 活动范围小于该阈值且几乎没有偏航变化时，
# 视为全程/近乎静止的退化 episode，直接整集丢弃而非压缩保留。
CLEANING_MIN_EPISODE_DISPLACEMENT_CM = 50.0


def limit_cv2_threads(cv2_module: Any) -> None:
    set_num_threads = getattr(cv2_module, "setNumThreads", None)
    if set_num_threads is not None:
        set_num_threads(RGB_DECODE_THREADS)

# UE 录制使用 +X 前、+Y 右、+Z 上；目标机体系要求 +Y 为左，因此只需要翻转 Y 轴。
UE_TO_TARGET = np.diag([1.0, -1.0, 1.0]).astype(np.float32)

# UE 相机局部轴为 +X 前、+Y 右、+Z 上；OpenCV 相机轴为 +X 右、+Y 下、+Z 前。
# 该矩阵把 OpenCV 相机坐标中的点转换到 UE 相机坐标。
UE_CAMERA_FROM_OPENCV = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Unreal Go2 recording episodes to LeRobot v2.1 format.")
    parser.add_argument("--raw_dir", type=str, required=True, help="UE OutputRoot, scene/user dir, or one episode_* dir.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory used to store scene-grouped exported datasets.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Deprecated. Kept for CLI compatibility; scene-grouped output uses output_dir directly.",
    )
    parser.add_argument("--camera_keys", type=str, default=",".join(DEFAULT_CAMERA_KEYS), help="Comma-separated cameras to export.")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of writer worker processes.")
    parser.add_argument("--codec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"], help="Video codec.")
    parser.add_argument("--pix_fmt", type=str, default="auto", choices=["auto", "yuv420p", "yuv444p"], help="Video pixel format.")
    parser.add_argument("--extrinsic_tolerance_translation_m", type=float, default=1e-4)
    parser.add_argument("--extrinsic_tolerance_rotation_deg", type=float, default=0.1)
    parser.add_argument(
        "--skip_invalid_episodes",
        action="store_true",
        help="Skip incompatible episodes and record them in meta/unreal_conversion_report.json.",
    )
    parser.add_argument(
        "--split_by_schema",
        action="store_true",
        help="Export one LeRobot dataset per fps/resolution schema.",
    )
    parser.add_argument(
        "--trim_extra_tail_frame",
        action="store_true",
        help="Trim one extra tail frame from frames.jsonl when it is exactly meta.frame_count + 1.",
    )
    parser.add_argument(
        "--skip_depth",
        action="store_true",
        help="Do not validate, decode, or export depth sidecar images.",
    )
    parser.add_argument(
        "--copy_rgb_mp4",
        action="store_true",
        help="Directly copy compatible source RGB MP4 files instead of decoding and re-encoding them.",
    )
    parser.add_argument(
        "--clean_invalid_data",
        action="store_true",
        help="Drop direction-invalid episodes and compress stuck-only episodes during conversion.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scene outputs that already completed successfully.",
    )
    parser.add_argument(
        "--overwrite_incomplete",
        action="store_true",
        help="With --resume, move incomplete scene outputs aside and rebuild instead of failing.",
    )
    parser.add_argument(
        "--scan_cache",
        type=str,
        default=None,
        help="Path to a pickle scan cache. Defaults to output_dir/unreal_scan_cache.pkl for --resume/--reuse_scan_cache.",
    )
    parser.add_argument(
        "--reuse_scan_cache",
        action="store_true",
        help="Load episode scan results from --scan_cache and skip per-episode validation.",
    )
    parser.add_argument(
        "--log_interval_seconds",
        type=float,
        default=30.0,
        help="Seconds between periodic scan/conversion progress log lines.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable per-episode debug logs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=json_default)
        file.write("\n")


def load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError as exc:
        logging.warning("Path existence check failed, treating as missing: %s (%s)", path, exc)
        return False


def scan_episode_dirs(raw_dir: str | Path) -> list[Path]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {root}")
    if root.is_file():
        raise ValueError(f"raw_dir must be a directory: {root}")

    if (root / "episode_meta.json").exists():
        return [root]

    return sorted(path.parent for path in root.rglob("episode_meta.json"))


def parse_camera_keys(value: str) -> list[str]:
    keys = [item.strip() for item in value.split(",") if item.strip()]
    if not keys:
        raise ValueError("camera_keys must not be empty.")
    return keys


def build_features(image_size: tuple[int, int], camera_keys: Iterable[str]) -> dict[str, dict[str, Any]]:
    height, width = image_size
    features: dict[str, dict[str, Any]] = {
        TASK_DESCRIPTION_KEY: {"dtype": "int32", "shape": (1,), "names": None},
        STATE_KEY: {"dtype": "float32", "shape": (7,), "names": {"axes": POSE_AXES}},
    }
    for camera_key in camera_keys:
        features[f"video.{camera_key}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    features[ACTION_KEY] = {"dtype": "float32", "shape": (7,), "names": {"axes": POSE_AXES}}
    return features


def episode_schema(meta: dict[str, Any]) -> tuple[int, tuple[int, int]]:
    fps = int(round(float(meta["sample_rate_hz"])))
    image_size = (int(meta["capture_height"]), int(meta["capture_width"]))
    return fps, image_size


def schema_suffix(schema: tuple[int, tuple[int, int]]) -> str:
    fps, (height, width) = schema
    return f"fps{fps}_{height}x{width}"


LEVEL_OK = "ok"
LEVEL_SOFT = "soft"
LEVEL_HARD = "hard"


@dataclass
class CleaningFrameDiag:
    frame_index: int
    level: str
    score: float
    angle_deg: float
    speed_cm_sec: float


@dataclass
class CleaningSegment:
    level: str
    start_index: int
    end_index: int
    start_frame: int
    end_frame: int
    peak_score: float = 0.0
    peak_angle_deg: float = 0.0


@dataclass
class CleaningStuckSegment:
    start_index: int
    end_index: int
    start_frame: int
    end_frame: int
    duration_sec: float = 0.0
    min_displacement_cm: float = 0.0


def cleaning_policy() -> dict[str, Any]:
    return {
        "min_speed_cm_sec": CLEANING_MIN_SPEED_CM_SEC,
        "deviation_angle_deg": CLEANING_DEVIATION_ANGLE_DEG,
        "time_constant": CLEANING_TIME_CONSTANT,
        "recovery_time_constant": CLEANING_RECOVERY_TIME_CONSTANT,
        "enter_soft": CLEANING_ENTER_SOFT,
        "enter_hard": CLEANING_ENTER_HARD,
        "exit_hard": CLEANING_EXIT_HARD,
        "exit_soft": CLEANING_EXIT_SOFT,
        "teleport_reset_cm": CLEANING_TELEPORT_RESET_CM,
        "max_violation_ratio": CLEANING_MAX_VIOLATION_RATIO,
        "hard_min_duration_sec": CLEANING_HARD_MIN_DURATION_SEC,
        "max_hard_ratio": CLEANING_MAX_HARD_RATIO,
        "max_stuck_ratio": CLEANING_MAX_STUCK_RATIO,
        "stuck_window_sec": CLEANING_STUCK_WINDOW_SEC,
        "stuck_max_displacement_cm": CLEANING_STUCK_MAX_DISPLACEMENT_CM,
        "stuck_keep_frames_per_segment": CLEANING_STUCK_KEEP_FRAMES_PER_SEGMENT,
        "stuck_boundary_keep_frames": CLEANING_STUCK_BOUNDARY_KEEP_FRAMES,
        "stuck_yaw_preserve_threshold_deg": CLEANING_STUCK_YAW_PRESERVE_THRESHOLD_DEG,
        "min_cleaned_frames": CLEANING_MIN_CLEANED_FRAMES,
        "min_episode_displacement_cm": CLEANING_MIN_EPISODE_DISPLACEMENT_CM,
    }


def frame_pose_xy_yaw(frame: dict[str, Any]) -> tuple[float, float, float] | None:
    pose = frame.get("pose")
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        return None
    try:
        return float(pose[0]), float(pose[1]), float(pose[5])
    except (TypeError, ValueError):
        return None


def frame_sim_time(frame: dict[str, Any], fallback_index: int) -> float:
    for key in ("timestamp_sim_sec", "timestamp_wall_sec", "timestamp"):
        value = frame.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return float(fallback_index) / 10.0


def shortest_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def yaw_delta_deg(a: float, b: float) -> float:
    return abs(shortest_angle_deg(b - a))


def forward2d_from_yaw(yaw_deg: float) -> tuple[float, float]:
    yaw = math.radians(yaw_deg)
    return math.cos(yaw), math.sin(yaw)


def next_cleaning_level(level: str, score: float) -> str:
    if level == LEVEL_OK:
        if score >= CLEANING_ENTER_HARD:
            return LEVEL_HARD
        if score >= CLEANING_ENTER_SOFT:
            return LEVEL_SOFT
        return LEVEL_OK
    if level == LEVEL_SOFT:
        if score >= CLEANING_ENTER_HARD:
            return LEVEL_HARD
        if score <= CLEANING_EXIT_SOFT:
            return LEVEL_OK
        return LEVEL_SOFT
    if score <= CLEANING_EXIT_HARD:
        return LEVEL_OK
    return LEVEL_HARD


def diagnose_direction(frames: list[dict[str, Any]]) -> tuple[list[CleaningFrameDiag | None], list[CleaningSegment]]:
    per_frame: list[CleaningFrameDiag | None] = []
    segments: list[CleaningSegment] = []
    last_x: float | None = None
    last_y: float | None = None
    last_time: float | None = None
    score = 0.0
    level = LEVEL_OK
    current_segment: CleaningSegment | None = None

    for index, frame in enumerate(frames):
        fidx = int(frame.get("frame_index", index))
        pose = frame_pose_xy_yaw(frame)
        if pose is None:
            per_frame.append(None)
            last_x = last_y = last_time = None
            if current_segment is not None:
                segments.append(current_segment)
                current_segment = None
            level = LEVEL_OK
            score = 0.0
            continue

        x, y, yaw = pose
        t = frame_sim_time(frame, index)
        angle_deg = 0.0
        speed = 0.0
        bad = False

        if last_x is not None and last_y is not None and last_time is not None:
            dx = x - last_x
            dy = y - last_y
            disp = math.hypot(dx, dy)
            dt = max(t - last_time, 1e-6)
            if disp > CLEANING_TELEPORT_RESET_CM:
                score = 0.0
                if current_segment is not None:
                    segments.append(current_segment)
                    current_segment = None
                level = LEVEL_OK
            else:
                speed = disp / dt
                if speed >= CLEANING_MIN_SPEED_CM_SEC:
                    fx, fy = forward2d_from_yaw(yaw)
                    vlen = math.hypot(dx, dy)
                    dot = (dx * fx + dy * fy) / max(vlen, 1e-9)
                    dot = max(-1.0, min(1.0, dot))
                    angle_deg = math.degrees(math.acos(dot))
                    bad = angle_deg > CLEANING_DEVIATION_ANGLE_DEG

                    tau = CLEANING_TIME_CONSTANT if bad else CLEANING_RECOVERY_TIME_CONSTANT
                    alpha = 1.0 - math.exp(-dt / max(tau, 1e-9))
                    target = 1.0 if bad else 0.0
                    score += alpha * (target - score)

        new_level = next_cleaning_level(level, score)
        if new_level != level:
            if current_segment is not None:
                segments.append(current_segment)
                current_segment = None
            if new_level != LEVEL_OK:
                current_segment = CleaningSegment(new_level, index, index, fidx, fidx, score, angle_deg)
        elif current_segment is not None:
            current_segment.end_index = index
            current_segment.end_frame = fidx
            current_segment.peak_score = max(current_segment.peak_score, score)
            current_segment.peak_angle_deg = max(current_segment.peak_angle_deg, angle_deg)

        level = new_level
        per_frame.append(CleaningFrameDiag(fidx, level, score, angle_deg, speed))
        last_x, last_y, last_time = x, y, t

    if current_segment is not None:
        segments.append(current_segment)
    return per_frame, segments


def detect_stuck_segments(frames: list[dict[str, Any]]) -> tuple[list[bool | None], list[CleaningStuckSegment]]:
    points: list[tuple[int, float, float, float] | None] = []
    for index, frame in enumerate(frames):
        pose = frame_pose_xy_yaw(frame)
        if pose is None:
            points.append(None)
        else:
            points.append((index, pose[0], pose[1], frame_sim_time(frame, index)))

    flags: list[bool | None] = [None] * len(frames)
    first_valid_t = next((item[3] for item in points if item is not None), None)
    left = 0
    for index, point in enumerate(points):
        if point is None:
            left = index + 1
            continue
        _, x, y, t = point
        while left < index and points[left] is not None and points[left][3] < t - CLEANING_STUCK_WINDOW_SEC:
            left += 1
        if left >= len(points) or points[left] is None:
            continue
        if first_valid_t is not None and t - first_valid_t < CLEANING_STUCK_WINDOW_SEC:
            flags[index] = False
            continue
        _, x0, y0, _ = points[left]
        flags[index] = math.hypot(x - x0, y - y0) < CLEANING_STUCK_MAX_DISPLACEMENT_CM

    segments: list[CleaningStuckSegment] = []
    current: CleaningStuckSegment | None = None
    current_start_index = 0
    for index, flag in enumerate(flags):
        if flag is True:
            fidx = int(frames[index].get("frame_index", index))
            if current is None:
                current = CleaningStuckSegment(index, index, fidx, fidx, 0.0, math.inf)
                current_start_index = index
            else:
                current.end_index = index
                current.end_frame = fidx
            point = points[index]
            start_point = points[current_start_index]
            if point is not None and start_point is not None:
                current.duration_sec = point[3] - start_point[3]
                min_disp = math.inf
                for j in range(current_start_index, index + 1):
                    p = points[j]
                    if p is None:
                        continue
                    min_disp = min(min_disp, math.hypot(point[1] - p[1], point[2] - p[2]))
                current.min_displacement_cm = min_disp
        elif current is not None:
            if math.isinf(current.min_displacement_cm):
                current.min_displacement_cm = 0.0
            segments.append(current)
            current = None
    if current is not None:
        if math.isinf(current.min_displacement_cm):
            current.min_displacement_cm = 0.0
        segments.append(current)
    return flags, segments


def judge_cleaning_validity(
    per_frame: list[CleaningFrameDiag | None],
    segments: list[CleaningSegment],
    stuck_flags: list[bool | None],
    fps: float,
) -> tuple[list[dict[str, Any]], bool]:
    moving = [item for item in per_frame if item is not None and item.speed_cm_sec > 0]
    bad_frames = [item for item in per_frame if item is not None and item.level != LEVEL_OK]
    hard_frames = [item for item in per_frame if item is not None and item.level == LEVEL_HARD]
    rules: list[dict[str, Any]] = []

    violation_ratio = len(bad_frames) / len(moving) if moving else 0.0
    if violation_ratio > CLEANING_MAX_VIOLATION_RATIO:
        rules.append(
            {
                "rule": "violation_ratio_exceeded",
                "value": round(violation_ratio, 4),
                "limit": CLEANING_MAX_VIOLATION_RATIO,
            }
        )

    hard_min_frames = CLEANING_HARD_MIN_DURATION_SEC * fps
    longest_hard = max(
        (segment.end_frame - segment.start_frame + 1 for segment in segments if segment.level == LEVEL_HARD),
        default=0,
    )
    if longest_hard >= hard_min_frames:
        rules.append(
            {
                "rule": "hard_segment_too_long",
                "longest_hard_frames": longest_hard,
                "limit_frames": round(hard_min_frames, 1),
            }
        )

    hard_ratio = len(hard_frames) / len(moving) if moving else 0.0
    if hard_ratio > CLEANING_MAX_HARD_RATIO:
        rules.append(
            {
                "rule": "hard_ratio_exceeded",
                "value": round(hard_ratio, 4),
                "limit": CLEANING_MAX_HARD_RATIO,
            }
        )

    stuck_count = sum(1 for item in stuck_flags if item is True)
    stuck_ratio = stuck_count / len(stuck_flags) if stuck_flags else 0.0
    stuck_rule_triggered = stuck_ratio > CLEANING_MAX_STUCK_RATIO
    if stuck_rule_triggered:
        rules.append(
            {
                "rule": "stuck_ratio_exceeded",
                "value": round(stuck_ratio, 4),
                "limit": CLEANING_MAX_STUCK_RATIO,
                "stuck_frames": stuck_count,
            }
        )
    return rules, stuck_rule_triggered


def segment_yaw_delta(frames: list[dict[str, Any]], start: int, end: int) -> float:
    total = 0.0
    previous: float | None = None
    for index in range(start, end + 1):
        pose = frame_pose_xy_yaw(frames[index])
        if pose is None:
            previous = None
            continue
        yaw = pose[2]
        if previous is not None:
            total += yaw_delta_deg(previous, yaw)
        previous = yaw
    return total


def evenly_spaced_indices(start: int, end: int, count: int) -> set[int]:
    """在 [start, end] 内取最多 count 个均匀分布的索引（始终包含首尾）。"""
    if end <= start:
        return {start}
    count = max(2, min(count, end - start + 1))
    return {start + round(step * (end - start) / (count - 1)) for step in range(count)}


def episode_xy_span_cm(frames: list[dict[str, Any]]) -> float | None:
    """整段轨迹 XY 位置包围盒的对角线长度（cm）。有效位姿不足 2 个时返回 None。"""
    xs: list[float] = []
    ys: list[float] = []
    for frame in frames:
        pose = frame_pose_xy_yaw(frame)
        if pose is not None:
            xs.append(pose[0])
            ys.append(pose[1])
    if len(xs) < 2:
        return None
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def compressed_stuck_keep_indices(
    frames: list[dict[str, Any]],
    stuck_flags: list[bool | None],
    stuck_segments: list[CleaningStuckSegment],
    fps: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    keep = {index for index, flag in enumerate(stuck_flags) if flag is not True}
    segment_reports: list[dict[str, Any]] = []

    for segment_index, segment in enumerate(stuck_segments):
        start = segment.start_index
        end = segment.end_index
        segment_indices = set(range(start, end + 1))
        yaw_delta = segment_yaw_delta(frames, start, end)
        preserve_all = yaw_delta > CLEANING_STUCK_YAW_PRESERVE_THRESHOLD_DEG
        if preserve_all:
            selected = segment_indices
        else:
            # 固定保留 N 个均匀分布的代表帧（含首尾），与段时长无关，
            # 这样无论静止持续多久，压缩后都不会残留长时间静止。
            selected = evenly_spaced_indices(start, end, CLEANING_STUCK_KEEP_FRAMES_PER_SEGMENT)
            for offset in range(1, CLEANING_STUCK_BOUNDARY_KEEP_FRAMES + 1):
                if start - offset >= 0:
                    selected.add(start - offset)
                if end + offset < len(frames):
                    selected.add(end + offset)
        keep.update(selected)
        segment_reports.append(
            {
                "segment_index": segment_index,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "start_index": start,
                "end_index": end,
                "original_frames": end - start + 1,
                "kept_frames": len(selected & segment_indices),
                "yaw_delta_deg": round(yaw_delta, 3),
                "preserved_all_due_to_yaw": preserve_all,
                "duration_sec": round(segment.duration_sec, 3),
                "min_displacement_cm": round(segment.min_displacement_cm, 3),
            }
        )
    return sorted(keep), segment_reports


def build_cleaning_decision(frames: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    per_frame, direction_segments = diagnose_direction(frames)
    stuck_flags, stuck_segments = detect_stuck_segments(frames)
    rules, stuck_rule_triggered = judge_cleaning_validity(per_frame, direction_segments, stuck_flags, fps)

    # 整集级退化静止判据：整段 XY 活动范围极小且几乎没有转向时，视为全程/近乎静止，
    # 直接整集丢弃（压缩这类集毫无价值，且 stuck 检测的预热豁免会漏判开头的静止帧）。
    episode_span_cm = episode_xy_span_cm(frames)
    episode_yaw_delta = segment_yaw_delta(frames, 0, len(frames) - 1) if frames else 0.0
    if (
        episode_span_cm is not None
        and episode_span_cm < CLEANING_MIN_EPISODE_DISPLACEMENT_CM
        and episode_yaw_delta <= CLEANING_STUCK_YAW_PRESERVE_THRESHOLD_DEG
    ):
        rules = rules + [
            {
                "rule": "static_episode",
                "value": round(episode_span_cm, 3),
                "limit": CLEANING_MIN_EPISODE_DISPLACEMENT_CM,
                "yaw_delta_deg": round(episode_yaw_delta, 3),
            }
        ]

    rule_names = {item["rule"] for item in rules}
    direction_rules = {
        "violation_ratio_exceeded",
        "hard_ratio_exceeded",
        "hard_segment_too_long",
    }
    stuck_frames = sum(1 for item in stuck_flags if item is True)
    base = {
        "cleaning_applied": True,
        "policy": cleaning_policy(),
        "original_frame_count": len(frames),
        "stuck_frames": stuck_frames,
        "stuck_segments": [
            {
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "start_index": segment.start_index,
                "end_index": segment.end_index,
                "duration_sec": round(segment.duration_sec, 3),
                "min_displacement_cm": round(segment.min_displacement_cm, 3),
            }
            for segment in stuck_segments
        ],
        "direction_segments": [
            {
                "level": segment.level,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "peak_score": round(segment.peak_score, 4),
                "peak_angle_deg": round(segment.peak_angle_deg, 2),
            }
            for segment in direction_segments
        ],
        "validity_rules": rules,
    }
    if "static_episode" in rule_names:
        return {
            **base,
            "decision": "drop_static_episode",
            "drop_reasons": ["static_episode"],
            "source_frame_indices": [],
            "cleaned_frame_count": 0,
            "dropped_stuck_frame_count": 0,
            "compressed_stuck_segments": [],
        }

    if rule_names & direction_rules:
        return {
            **base,
            "decision": "drop_direction_invalid",
            "drop_reasons": sorted(rule_names & direction_rules),
            "source_frame_indices": [],
            "cleaned_frame_count": 0,
            "dropped_stuck_frame_count": 0,
            "compressed_stuck_segments": [],
        }

    if stuck_rule_triggered:
        keep_indices, segment_reports = compressed_stuck_keep_indices(frames, stuck_flags, stuck_segments, fps)
        if len(keep_indices) < CLEANING_MIN_CLEANED_FRAMES:
            return {
                **base,
                "decision": "drop_empty_after_clean",
                "drop_reasons": ["cleaned_frame_count_below_minimum"],
                "source_frame_indices": keep_indices,
                "cleaned_frame_count": len(keep_indices),
                "dropped_stuck_frame_count": len(frames) - len(keep_indices),
                "compressed_stuck_segments": segment_reports,
            }
        return {
            **base,
            "decision": "keep_compressed",
            "drop_reasons": [],
            "source_frame_indices": keep_indices,
            "cleaned_frame_count": len(keep_indices),
            "dropped_stuck_frame_count": len(frames) - len(keep_indices),
            "compressed_stuck_segments": segment_reports,
        }

    return {
        **base,
        "decision": "keep",
        "drop_reasons": [],
        "source_frame_indices": list(range(len(frames))),
        "cleaned_frame_count": len(frames),
        "dropped_stuck_frame_count": 0,
        "compressed_stuck_segments": [],
    }


def normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion)
    if norm <= 0:
        raise ValueError("Quaternion norm must be positive.")
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion.astype(np.float32)


def transform_to_pose_vector(transform: Any) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {transform.shape}")
    translation = transform[:3, 3]
    quaternion = normalize_quaternion_xyzw(Rotation.from_matrix(transform[:3, :3]).as_quat().astype(np.float32))
    return np.concatenate([translation, quaternion], axis=0).astype(np.float32)


def select_video_pixel_format(image_size: tuple[int, int], codec: str, pix_fmt: str) -> str:
    if pix_fmt != "auto":
        return pix_fmt
    if codec in {"h264", "hevc"} and any(size % 2 != 0 for size in image_size):
        logging.warning(
            "Image size %s is not divisible by 2, using yuv444p to keep the original resolution.",
            image_size,
        )
        return "yuv444p"
    return "yuv420p"


def decode_hue_depth_rgb(
    rgb: np.ndarray,
    min_meters: float,
    max_meters: float,
    *,
    dark_threshold: int = DEPTH_DARK_THRESHOLD,
    saturation_threshold: int = DEPTH_SATURATION_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode UE HueMp4 RGB pixels into uint16 millimeter depth.

    UE maps valid depth linearly to HSV hue 0..300 degrees with full saturation
    and value. Invalid depth is black. Lossy H.264 can perturb those ideal
    colors, so dark or nearly gray pixels are treated as invalid.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")
    if max_meters <= min_meters:
        raise ValueError(f"Invalid hue depth range: min={min_meters} max={max_meters}")

    values = rgb.astype(np.float32)
    red = values[..., 0]
    green = values[..., 1]
    blue = values[..., 2]
    maximum = values.max(axis=2)
    minimum = values.min(axis=2)
    delta = maximum - minimum
    valid = (maximum > float(dark_threshold)) & (delta > float(saturation_threshold))

    safe_delta = np.where(valid, delta, 1.0)
    hue = np.zeros(maximum.shape, dtype=np.float32)
    red_max = valid & (red >= green) & (red >= blue)
    green_max = valid & ~red_max & (green >= blue)
    blue_max = valid & ~red_max & ~green_max

    hue[red_max] = 60.0 * np.mod(
        (green[red_max] - blue[red_max]) / safe_delta[red_max],
        6.0,
    )
    hue[green_max] = 60.0 * (
        (blue[green_max] - red[green_max]) / safe_delta[green_max] + 2.0
    )
    hue[blue_max] = 60.0 * (
        (red[blue_max] - green[blue_max]) / safe_delta[blue_max] + 4.0
    )
    hue = np.clip(hue, 0.0, 300.0)

    depth_meters = float(min_meters) + hue / 300.0 * (float(max_meters) - float(min_meters))
    depth_mm_float = np.clip(depth_meters * 1000.0, 0.0, float(np.iinfo(np.uint16).max))
    depth_mm = np.rint(depth_mm_float).astype(np.uint16)
    depth_mm[~valid] = 0
    return depth_mm, valid


def homogeneous_inv(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected shape (4, 4), got {transform.shape}")
    inverse = np.eye(4, dtype=transform.dtype)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def unreal_pose_to_target_transform(pose: list[float] | np.ndarray) -> np.ndarray:
    """将 UE 的 [cm, Roll/Pitch/Yaw] 位姿转换到目标机体系坐标约定下的 SE(3)。"""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Unreal pose must have shape (6,), got {pose.shape}")

    location_m = pose[:3] / 100.0
    roll, pitch, yaw = pose[3:]
    rotation_ue = Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix()

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = (UE_TO_TARGET @ rotation_ue @ UE_TO_TARGET).astype(np.float32)
    transform[:3, 3] = (UE_TO_TARGET @ location_m).astype(np.float32)
    return transform


def unreal_camera_pose_to_target_opencv_transform(pose: list[float] | np.ndarray) -> np.ndarray:
    """将 UE 相机 world pose 转为目标 world 下的 OpenCV 相机坐标系 pose。"""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Unreal camera pose must have shape (6,), got {pose.shape}")

    location_m = pose[:3] / 100.0
    roll, pitch, yaw = pose[3:]
    rotation_ue = Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix()

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = (UE_TO_TARGET @ rotation_ue @ UE_CAMERA_FROM_OPENCV).astype(np.float32)
    transform[:3, 3] = (UE_TO_TARGET @ location_m).astype(np.float32)
    return transform


def body_from_camera_for_frame(frame: dict[str, Any], camera_key: str) -> np.ndarray:
    # 外参定义为 T_body<-camera，即把 OpenCV 相机坐标中的点变换到目标机体系。
    body_transform = unreal_pose_to_target_transform(frame["pose"])
    camera_transform = unreal_camera_pose_to_target_opencv_transform(frame[f"camera_pose_{camera_key}"])
    return (homogeneous_inv(body_transform) @ camera_transform).astype(np.float32)


def intrinsic_4(frame_or_meta: dict[str, Any], camera_key: str) -> list[float]:
    """从 UE 写出的 3x3 K 展平数组中提取 [fx, fy, cx, cy]。"""
    key = f"K_{camera_key}"
    if key not in frame_or_meta:
        raise ValueError(f"Missing {key}")
    matrix = frame_or_meta[key]
    if len(matrix) != 9:
        raise ValueError(f"{key} must contain 9 values, got {len(matrix)}")
    return [float(matrix[0]), float(matrix[4]), float(matrix[2]), float(matrix[5])]


def intrinsic_matrix(frame_or_meta: dict[str, Any], camera_key: str) -> list[list[float]]:
    """从 UE 写出的 3x3 K 展平数组中恢复完整内参矩阵。"""
    key = f"K_{camera_key}"
    if key not in frame_or_meta:
        raise ValueError(f"Missing {key}")
    matrix = frame_or_meta[key]
    if len(matrix) != 9:
        raise ValueError(f"{key} must contain 9 values, got {len(matrix)}")
    values = [float(value) for value in matrix]
    return [values[0:3], values[3:6], values[6:9]]


def rotation_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = Rotation.from_matrix(a[:3, :3].T @ b[:3, :3])
    return float(np.degrees(delta.magnitude()))


def validate_fixed_extrinsics(
    episode_dir: Path,
    frames: list[dict[str, Any]],
    camera_keys: list[str],
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
) -> dict[str, np.ndarray]:
    """严格校验相机安装外参在一个 episode 内保持不变。"""
    if not frames:
        raise ValueError(f"No frames in {episode_dir}")

    baseline = {camera: body_from_camera_for_frame(frames[0], camera) for camera in camera_keys}
    max_translation: dict[str, float] = {camera: 0.0 for camera in camera_keys}
    max_rotation: dict[str, float] = {camera: 0.0 for camera in camera_keys}

    for frame in frames[1:]:
        for camera in camera_keys:
            current = body_from_camera_for_frame(frame, camera)
            trans_delta = float(np.linalg.norm(current[:3, 3] - baseline[camera][:3, 3]))
            rot_delta = rotation_delta_deg(baseline[camera], current)
            max_translation[camera] = max(max_translation[camera], trans_delta)
            max_rotation[camera] = max(max_rotation[camera], rot_delta)

    violations = [
        f"{camera}: translation={max_translation[camera]:.6g}m rotation={max_rotation[camera]:.6g}deg"
        for camera in camera_keys
        if max_translation[camera] > translation_tolerance_m or max_rotation[camera] > rotation_tolerance_deg
    ]
    if violations:
        raise ValueError(f"Dynamic body_from_camera in {episode_dir}: " + "; ".join(violations))

    return baseline


def load_task_info(episode_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    """Read UE subtask segments and return the first non-empty fallback task."""
    path = episode_dir / "task_info.csv"
    if not path.exists():
        return "", []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        first_line = file.readline()
        if not first_line.startswith("sep="):
            file.seek(0)
        reader = csv.DictReader(file)
        for row in reader:
            if not row:
                continue
            parsed = dict(row)
            for key in ("subtask_index", "start_frame", "end_frame"):
                if parsed.get(key) not in (None, ""):
                    parsed[key] = int(parsed[key])
            rows.append(parsed)

    task = next((str(row.get("name", "")).strip() for row in rows if str(row.get("name", "")).strip()), "")
    return task, rows


def resolve_frame_tasks(
    task_info: list[dict[str, Any]],
    frame_count: int,
    fallback_task: str,
) -> tuple[list[str], dict[str, Any]]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if not task_info:
        return [fallback_task] * frame_count, {"status": "no_task_info"}

    try:
        starts = [int(row["start_frame"]) for row in task_info]
        ends = [int(row["end_frame"]) for row in task_info]
        if starts[0] != 0:
            raise ValueError(f"first start_frame must be 0, got {starts[0]}")
        if any(current <= previous for previous, current in zip(starts, starts[1:])):
            raise ValueError(f"start_frame values must be strictly increasing: {starts}")
        for index in range(len(task_info) - 1):
            if ends[index] != starts[index + 1]:
                raise ValueError(
                    f"segment {index} end_frame={ends[index]} does not match "
                    f"next start_frame={starts[index + 1]}"
                )
        if ends[-1] != frame_count - 1:
            raise ValueError(
                f"last end_frame must equal final frame index {frame_count - 1}, got {ends[-1]}"
            )

        tasks = [""] * frame_count
        for index, row in enumerate(task_info):
            start = starts[index]
            stop = starts[index + 1] if index + 1 < len(starts) else ends[index] + 1
            if start < 0 or stop > frame_count or stop <= start:
                raise ValueError(f"invalid segment bounds [{start}, {stop})")
            name = str(row.get("name", "")).strip()
            tasks[start:stop] = [name] * (stop - start)
        return tasks, {"status": "mapped", "num_segments": len(task_info)}
    except (KeyError, TypeError, ValueError) as exc:
        return (
            [fallback_task] * frame_count,
            {
                "status": "fallback",
                "reason": str(exc),
                "fallback_task": fallback_task,
            },
        )


def infer_source_ids(episode_dir: Path) -> tuple[str, str]:
    user_id = episode_dir.parent.name if episode_dir.parent else ""
    scene_id = episode_dir.parent.parent.name if episode_dir.parent and episode_dir.parent.parent else ""
    return scene_id, user_id


def load_media_meta(episode_dir: Path, modality: str) -> dict[str, Any]:
    path = episode_dir / modality / "meta.json"
    if not path.exists():
        raise ValueError(f"missing {modality}/meta.json")
    meta = load_json(path)
    if not isinstance(meta, dict):
        raise ValueError(f"{modality}/meta.json must contain a JSON object")
    return meta


def validate_media_meta(
    episode_dir: Path,
    episode_meta: dict[str, Any],
    rgb_meta: dict[str, Any],
    depth_meta: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recover legacy episode fields and cross-check the two media manifests."""
    recovered = dict(episode_meta)
    repairs: list[dict[str, Any]] = []
    source_scene_id, _ = infer_source_ids(episode_dir)
    if not str(recovered.get("scene_id", "")).strip():
        recovered["scene_id"] = source_scene_id
        repairs.append(
            {
                "field": "scene_id",
                "action": "recovered_from_directory",
                "value": source_scene_id,
            }
        )

    rgb_cameras = tuple(str(item) for item in (rgb_meta.get("camera_names") or []))
    if not rgb_cameras:
        raise ValueError("rgb media metadata must declare camera_names")
    if depth_meta is not None:
        depth_cameras = tuple(str(item) for item in (depth_meta.get("camera_names") or []))
        if not depth_cameras:
            raise ValueError("depth media metadata must declare camera_names")
        if set(rgb_cameras) != set(depth_cameras):
            raise ValueError(
                f"RGB/Depth camera_names mismatch: rgb={list(rgb_cameras)} depth={list(depth_cameras)}"
            )

    declared_cameras = tuple(str(item) for item in (recovered.get("camera_names") or []))
    if not declared_cameras:
        recovered["camera_names"] = list(rgb_cameras)
        repairs.append(
            {
                "field": "camera_names",
                "action": "recovered_from_media_metadata",
                "value": list(rgb_cameras),
            }
        )
    elif set(declared_cameras) != set(rgb_cameras):
        raise ValueError(
            f"episode/media camera_names mismatch: episode={list(declared_cameras)} "
            f"media={list(rgb_cameras)}"
        )

    expected_width = int(recovered["capture_width"])
    expected_height = int(recovered["capture_height"])
    expected_fps = float(recovered["sample_rate_hz"])
    media_metas: list[tuple[str, dict[str, Any]]] = [("rgb", rgb_meta)]
    if depth_meta is not None:
        media_metas.append(("depth", depth_meta))
    for modality, media_meta in media_metas:
        width = int(media_meta.get("capture_width", -1))
        height = int(media_meta.get("capture_height", -1))
        fps = float(media_meta.get("frame_rate_hz", -1))
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"{modality}/meta.json resolution mismatch: "
                f"episode={expected_width}x{expected_height} media={width}x{height}"
            )
        if not np.isclose(fps, expected_fps, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"{modality}/meta.json frame rate mismatch: episode={expected_fps} media={fps}"
            )

    return recovered, repairs


def find_undeclared_media(
    episode_dir: Path,
    media_meta: dict[str, Any],
    modality: str,
) -> list[dict[str, Any]]:
    storage = str(media_meta.get("storage", "")).strip().lower()
    warnings: list[dict[str, Any]] = []
    for camera in media_meta.get("camera_names") or []:
        video_path = episode_dir / modality / f"{camera}.mp4"
        image_dir = episode_dir / modality / str(camera)
        png_count = len(list(image_dir.glob("*.png"))) if image_dir.is_dir() else 0
        if storage == "mp4" and png_count:
            warnings.append(
                {
                    "source_episode_path": str(episode_dir),
                    "stage": "media_selection",
                    "modality": modality,
                    "camera": str(camera),
                    "warning": "ignored_undeclared_png_files",
                    "count": png_count,
                }
            )
        if storage == "png_sequence" and video_path.exists():
            warnings.append(
                {
                    "source_episode_path": str(episode_dir),
                    "stage": "media_selection",
                    "modality": modality,
                    "camera": str(camera),
                    "warning": "ignored_undeclared_mp4",
                }
            )
    return warnings


@dataclass
class CameraImageSource:
    video_path: Path | None
    image_paths: list[Path] | None
    frame_count: int
    source_frame_indices: list[int] | None = None

    @classmethod
    def from_episode(
        cls,
        episode_dir: Path,
        rgb_meta: dict[str, Any],
        camera_key: str,
        frame_count: int,
        allow_extra_tail_frame: bool = False,
        source_frame_indices: list[int] | None = None,
        media_frame_count: int | None = None,
    ) -> "CameraImageSource":
        storage = str(rgb_meta.get("storage", "")).strip().lower()
        video_path = episode_dir / "rgb" / f"{camera_key}.mp4"
        image_dir = episode_dir / "rgb" / camera_key
        image_paths = sorted(image_dir.glob("*.png")) if image_dir.is_dir() else []
        expected_media_frames = int(media_frame_count if media_frame_count is not None else frame_count)
        selected_indices = list(source_frame_indices) if source_frame_indices is not None else None

        if storage == "mp4":
            if image_paths:
                logging.warning(
                    "Ignoring %d undeclared RGB PNG files because rgb/meta.json selects mp4: %s camera=%s",
                    len(image_paths),
                    episode_dir,
                    camera_key,
                )
            if not path_exists(video_path):
                raise ValueError(f"Missing RGB MP4 selected by rgb/meta.json: {video_path}")
            import cv2

            capture = cv2.VideoCapture(str(video_path))
            try:
                if not capture.isOpened():
                    raise ValueError(f"Failed to open RGB video: {video_path}")
                encoded_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                capture.release()
            accepted_counts = {expected_media_frames}
            if allow_extra_tail_frame:
                accepted_counts.add(expected_media_frames + 1)
            if encoded_count > 0 and encoded_count not in accepted_counts:
                raise ValueError(
                    f"RGB video frame count mismatch for {episode_dir} camera {camera_key}: "
                    f"expected {expected_media_frames}, got {encoded_count}"
                )
            return cls(video_path=video_path, image_paths=None, frame_count=frame_count, source_frame_indices=selected_indices)

        if storage == "png_sequence":
            if path_exists(video_path):
                logging.warning(
                    "Ignoring undeclared RGB MP4 because rgb/meta.json selects png_sequence: %s",
                    video_path,
                )
            if len(image_paths) != expected_media_frames:
                raise ValueError(
                    f"RGB frame count mismatch for {episode_dir} camera {camera_key}: "
                    f"expected {expected_media_frames}, got {len(image_paths)}"
                )
            return cls(video_path=None, image_paths=image_paths, frame_count=frame_count, source_frame_indices=selected_indices)

        raise ValueError(f"Unsupported RGB storage {storage!r} in {episode_dir / 'rgb' / 'meta.json'}")

    def iter_rgb(self):
        indices = self.source_frame_indices or list(range(self.frame_count))
        if self.video_path is not None:
            import cv2

            limit_cv2_threads(cv2)
            capture = cv2.VideoCapture(str(self.video_path))
            if not capture.isOpened():
                raise ValueError(f"Failed to open video: {self.video_path}")
            try:
                target_set = set(indices)
                max_index = max(indices, default=-1)
                for index in range(max_index + 1):
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Video ended early at frame {index}: {self.video_path}")
                    if index in target_set:
                        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            finally:
                capture.release()
            return

        assert self.image_paths is not None
        for index in indices:
            path = self.image_paths[index]
            with Image.open(path) as image:
                yield np.asarray(image.convert("RGB"))


class UnrealEpisode:
    def __init__(
        self,
        episode_dir: Path,
        meta: dict[str, Any],
        frames: list[dict[str, Any]],
        camera_keys: list[str],
        task: str,
        task_idx: int,
        task_info: list[dict[str, Any]],
        body_from_camera: dict[str, np.ndarray],
        rgb_meta: dict[str, Any] | None = None,
        depth_meta: dict[str, Any] | None = None,
        frame_tasks: list[str] | None = None,
        task_indices: dict[str, int] | None = None,
        task_mapping: dict[str, Any] | None = None,
        export_depth: bool = True,
        copy_rgb_mp4: bool = False,
        source_frame_indices: list[int] | None = None,
        original_frame_count: int | None = None,
    ):
        self.episode_dir = episode_dir
        self.meta = meta
        self.frames = frames
        self.camera_keys = camera_keys
        self.task = task
        self.task_idx = task_idx
        self.task_info = task_info
        self.body_from_camera = body_from_camera
        self.rgb_meta = dict(rgb_meta) if rgb_meta is not None else load_media_meta(episode_dir, "rgb")
        self.export_depth = export_depth
        self.depth_meta = (
            dict(depth_meta)
            if depth_meta is not None
            else (load_media_meta(episode_dir, "depth") if export_depth else {})
        )
        self.depth_decode_stats: dict[str, Any] = {}
        self.allow_extra_tail_frame = bool(self.meta.get("_trimmed_extra_tail_frame", False))
        self.frame_tasks = list(frame_tasks) if frame_tasks is not None else [task] * len(frames)
        self.task_indices = dict(task_indices or {task: task_idx})
        self.task_mapping = dict(task_mapping or {"status": "legacy_single_task"})
        self.source_frame_indices = list(source_frame_indices) if source_frame_indices is not None else list(range(len(frames)))
        self.original_frame_count = int(original_frame_count if original_frame_count is not None else len(self.source_frame_indices))
        self.has_frame_filter = self.source_frame_indices != list(range(len(self.frames)))
        self.copy_rgb_mp4 = copy_rgb_mp4 and not self.has_frame_filter
        self.progress_reporter = None
        if len(self.frame_tasks) != len(self.frames):
            raise ValueError(
                f"frame task count mismatch: frames={len(self.frames)} tasks={len(self.frame_tasks)}"
            )
        if len(self.source_frame_indices) != len(self.frames):
            raise ValueError(
                f"source frame index count mismatch: frames={len(self.frames)} "
                f"indices={len(self.source_frame_indices)}"
            )
        self.image_sources = {
            camera: CameraImageSource.from_episode(
                episode_dir,
                self.rgb_meta,
                camera,
                len(frames),
                allow_extra_tail_frame=self.allow_extra_tail_frame,
                source_frame_indices=self.source_frame_indices,
                media_frame_count=self.original_frame_count,
            )
            for camera in camera_keys
        }

    def set_progress_reporter(self, reporter):
        self.progress_reporter = reporter

    def report_progress(self, stage: str, frames: int, total_frames: int, camera: str | None = None):
        if self.progress_reporter is None:
            return
        self.progress_reporter(
            {
                "stage": stage,
                "frames": frames,
                "total_frames": total_frames,
                "camera": camera,
            }
        )

    @property
    def direct_video_paths(self) -> dict[str, str]:
        if not self.copy_rgb_mp4 or self.allow_extra_tail_frame:
            return {}
        paths: dict[str, str] = {}
        for camera in self.camera_keys:
            source = self.image_sources[camera]
            if source.video_path is not None:
                paths[f"video.{camera}"] = str(source.video_path)
        return paths

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def metadata(self) -> dict[str, Any]:
        scene_id, user_id = infer_source_ids(self.episode_dir)
        metadata: dict[str, Any] = {
            "source_episode_path": str(self.episode_dir),
            "scene_id": scene_id,
            "user_id": user_id,
            "original_episode_index": int(self.meta.get("episode_index", -1)),
            "map_name": self.meta.get("map_name", ""),
            "frame_count": len(self.frames),
            "fps": int(round(float(self.meta.get("sample_rate_hz", 0)))),
            "capture_width": int(self.meta.get("capture_width", 0)),
            "capture_height": int(self.meta.get("capture_height", 0)),
            "camera_keys": self.camera_keys,
            "task": self.task,
            "task_info": self.task_info,
            "task_mapping": self.task_mapping,
            "created_at": self.meta.get("created_at", ""),
            "updated_at": self.meta.get("updated_at", ""),
            "rgb_media_meta": self.rgb_meta,
            "depth_decode_stats": self.depth_decode_stats,
        }
        if self.meta.get("_cleaning"):
            metadata["cleaning"] = self.meta["_cleaning"]
        if self.export_depth:
            metadata["depth_media_meta"] = self.depth_meta
            metadata["depth_output_format"] = "uint16_mm_png"
        for camera in self.camera_keys:
            video_key = f"video.{camera}"
            metadata[f"{video_key}.K"] = intrinsic_4(self.frames[0], camera)
            metadata[f"{video_key}.body_from_camera"] = self.body_from_camera[camera]
            metadata[f"K_{camera}"] = intrinsic_matrix(self.frames[0], camera)
            metadata[f"Extrinsic_{camera}"] = self.body_from_camera[camera]
        return metadata

    def prepare_episode(self, output_root: Path) -> dict[str, Any]:
        if not self.export_depth:
            return {}
        storage = str(self.depth_meta.get("storage", "")).strip().lower()
        encoding = str(self.depth_meta.get("video_encoding", "")).strip().lower()
        unit = str(self.depth_meta.get("depth_unit", "")).strip().lower()
        if storage != "mp4" or encoding != "huemp4":
            raise ValueError(
                f"Unsupported depth media format for {self.episode_dir}: "
                f"storage={storage!r} video_encoding={encoding!r}; expected HueMp4"
            )
        if unit != "millimeter":
            raise ValueError(f"Unsupported depth unit {unit!r} in {self.episode_dir / 'depth' / 'meta.json'}")

        min_meters = float(self.depth_meta["hue_min_meters"])
        max_meters = float(self.depth_meta["hue_max_meters"])
        staging_parent = Path(output_root) / ".unreal_depth_staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(tempfile.mkdtemp(prefix="episode_", dir=staging_parent))
        prepared: dict[str, Any] = {
            "staging_root": staging_root,
            "camera_dirs": {},
            "committed_dirs": [],
        }
        stats: dict[str, Any] = {}

        try:
            import cv2

            limit_cv2_threads(cv2)
            total_depth_frames = len(self.frames) * len(self.camera_keys)
            decoded_depth_frames = 0
            for camera in self.camera_keys:
                video_path = self.episode_dir / "depth" / f"{camera}.mp4"
                if not path_exists(video_path):
                    raise ValueError(f"Missing HueMp4 depth video: {video_path}")

                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    capture.release()
                    raise ValueError(f"Failed to open HueMp4 depth video: {video_path}")

                camera_dir = staging_root / camera
                camera_dir.mkdir(parents=True)
                valid_pixels = 0
                total_pixels = 0
                min_valid_mm: int | None = None
                max_valid_mm: int | None = None
                try:
                    target_set = set(self.source_frame_indices)
                    max_source_frame = max(self.source_frame_indices, default=-1)
                    output_frame_index = 0
                    for source_frame_index in range(max_source_frame + 1):
                        ok, bgr = capture.read()
                        if not ok:
                            raise ValueError(
                                f"Depth video ended early for {self.episode_dir} camera {camera}: "
                                f"expected source frame {source_frame_index}"
                            )
                        if source_frame_index not in target_set:
                            continue
                        rgb = bgr[..., ::-1]
                        depth_mm, valid = decode_hue_depth_rgb(rgb, min_meters, max_meters)
                        Image.fromarray(depth_mm).save(camera_dir / f"{output_frame_index:05d}.png")
                        output_frame_index += 1
                        decoded_depth_frames += 1
                        self.report_progress(
                            "depth_decode",
                            decoded_depth_frames,
                            total_depth_frames,
                            camera,
                        )
                        frame_valid = int(valid.sum())
                        valid_pixels += frame_valid
                        total_pixels += int(valid.size)
                        if frame_valid:
                            valid_depth = depth_mm[valid]
                            frame_min = int(valid_depth.min())
                            frame_max = int(valid_depth.max())
                            min_valid_mm = frame_min if min_valid_mm is None else min(min_valid_mm, frame_min)
                            max_valid_mm = frame_max if max_valid_mm is None else max(max_valid_mm, frame_max)

                    if output_frame_index != len(self.frames):
                        raise ValueError(
                            f"Depth frame selection mismatch for {self.episode_dir} camera {camera}: "
                            f"expected {len(self.frames)}, wrote {output_frame_index}"
                        )
                    for _source_frame_index in range(max_source_frame + 1, self.original_frame_count):
                        ok, _ = capture.read()
                        if not ok:
                            raise ValueError(
                                f"Depth video ended early for {self.episode_dir} camera {camera}: "
                                f"expected {self.original_frame_count} source frames"
                            )

                    ok, _ = capture.read()
                    if ok and self.allow_extra_tail_frame:
                        ok, _ = capture.read()
                    if ok:
                        raise ValueError(
                            f"Depth video frame count mismatch for {self.episode_dir} camera {camera}: "
                            f"expected {len(self.frames)}"
                            f"{' or one compatible tail frame' if self.allow_extra_tail_frame else ''}, "
                            "video has more"
                        )
                finally:
                    capture.release()

                prepared["camera_dirs"][camera] = camera_dir
                stats[camera] = {
                    "frame_count": len(self.frames),
                    "valid_pixel_count": valid_pixels,
                    "total_pixel_count": total_pixels,
                    "invalid_pixel_ratio": (
                        float(total_pixels - valid_pixels) / float(total_pixels) if total_pixels else 1.0
                    ),
                    "min_valid_depth_mm": min_valid_mm,
                    "max_valid_depth_mm": max_valid_mm,
                }
        except Exception:
            shutil.rmtree(staging_root, ignore_errors=True)
            try:
                staging_parent.rmdir()
            except OSError:
                pass
            raise

        self.depth_decode_stats = stats
        return prepared

    def commit_prepared_episode(
        self,
        output_root: Path,
        episode_index: int,
        prepared: dict[str, Any],
    ):
        if not prepared:
            return
        chunk = int(episode_index) // 1000
        for camera in self.camera_keys:
            source_dir = Path(prepared["camera_dirs"][camera])
            target_dir = (
                Path(output_root)
                / "images"
                / f"chunk-{chunk:03d}"
                / f"observation.depth.{camera}"
                / f"episode_{int(episode_index):06d}"
            )
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            if target_dir.exists():
                raise FileExistsError(f"Depth sidecar output already exists: {target_dir}")
            shutil.move(str(source_dir), str(target_dir))
            prepared["committed_dirs"].append(target_dir)
        staging_root = Path(prepared["staging_root"])
        shutil.rmtree(staging_root, ignore_errors=True)
        try:
            staging_root.parent.rmdir()
        except OSError:
            pass

    def discard_prepared_episode(self, prepared: dict[str, Any]):
        if not prepared:
            return
        staging_root = Path(prepared["staging_root"])
        shutil.rmtree(staging_root, ignore_errors=True)
        try:
            staging_root.parent.rmdir()
        except OSError:
            pass
        for path in prepared.get("committed_dirs", []):
            shutil.rmtree(Path(path), ignore_errors=True)

    def __iter__(self):
        direct_video_keys = set(self.direct_video_paths)
        image_iters = {
            camera: self.image_sources[camera].iter_rgb()
            for camera in self.camera_keys
            if f"video.{camera}" not in direct_video_keys
        }
        first_body_inv: np.ndarray | None = None

        for frame_index, frame in enumerate(self.frames):
            world_from_body = unreal_pose_to_target_transform(frame["pose"])
            if first_body_inv is None:
                # 按数据规范，trajectory 的 world 取第一帧机体坐标系。
                first_body_inv = homogeneous_inv(world_from_body)
            local_pose = transform_to_pose_vector((first_body_inv @ world_from_body).astype(np.float32))

            frame_task = self.frame_tasks[frame_index]
            item: dict[str, Any] = {
                TASK_DESCRIPTION_KEY: np.array([self.task_indices[frame_task]], dtype=np.int32),
                STATE_KEY: local_pose,
                ACTION_KEY: local_pose.copy(),
            }
            for camera in self.camera_keys:
                if camera in image_iters:
                    item[f"video.{camera}"] = next(image_iters[camera])
            yield item, frame_task


class UnrealEpisodeCollection:
    ROBOT_TYPE = "go2"
    INSTRUCTION_KEY = TASK_DESCRIPTION_KEY

    def __init__(
        self,
        raw_dir: str | Path,
        camera_keys: list[str],
        get_task_idx,
        translation_tolerance_m: float,
        rotation_tolerance_deg: float,
        skip_invalid_episodes: bool = False,
        target_schema: tuple[int, tuple[int, int]] | None = None,
        keep_all_schemas: bool = False,
        trim_extra_tail_frame: bool = False,
        initial_episodes: list[tuple] | None = None,
        initial_failures: list[dict[str, Any]] | None = None,
        initial_repairs: list[dict[str, Any]] | None = None,
        initial_exclusions: list[dict[str, Any]] | None = None,
        initial_warnings: list[dict[str, Any]] | None = None,
        export_depth: bool = True,
        copy_rgb_mp4: bool = False,
        log_interval_seconds: float = 30.0,
        clean_invalid_data: bool = False,
    ):
        self.raw_dir = Path(raw_dir)
        self.camera_keys = camera_keys
        self.get_task_idx = get_task_idx
        self.translation_tolerance_m = translation_tolerance_m
        self.rotation_tolerance_deg = rotation_tolerance_deg
        self.skip_invalid_episodes = skip_invalid_episodes
        self.target_schema = target_schema
        self.keep_all_schemas = keep_all_schemas
        self.trim_extra_tail_frame = trim_extra_tail_frame
        self.export_depth = export_depth
        self.copy_rgb_mp4 = copy_rgb_mp4
        self.clean_invalid_data = clean_invalid_data
        self.log_interval_seconds = max(1.0, float(log_interval_seconds))
        self.failed_episodes: list[dict[str, Any]] = list(initial_failures or [])
        self.repaired_episodes: list[dict[str, Any]] = list(initial_repairs or [])
        self.excluded_episodes: list[dict[str, Any]] = list(initial_exclusions or [])
        self.warnings: list[dict[str, Any]] = list(initial_warnings or [])
        self.prepared_episodes: list[dict[str, Any]] = []
        self.successful_episodes: list[dict[str, Any]] = []
        self.schema_groups: dict[str, dict[str, Any]] = {}
        self.schema_valid_episodes: list[tuple] = []
        self.episodes = list(initial_episodes) if initial_episodes is not None else self._load_episodes()

        if not self.episodes:
            if self.skip_invalid_episodes:
                self.fps = 0
                self.image_size = (0, 0)
                self.FEATURES = {}
                return
            raise ValueError(f"No completed Unreal episodes found under {self.raw_dir}")

        schema_candidates: list[tuple[tuple[int, tuple[int, int]], tuple]] = []
        for episode in self.episodes:
            episode_dir, meta, frames = episode[:3]
            try:
                frames = self._repair_frames_if_needed(episode_dir, meta, frames)
                frames = self._clean_frames_if_needed(episode_dir, meta, frames)
                if frames is None:
                    continue
                episode = (episode_dir, meta, frames, *episode[3:])
                schema_candidates.append((episode_schema(meta), episode))
            except Exception as exc:
                self._record_failure(episode_dir, "schema_validation", exc)

        self.schema_valid_episodes = [episode for _, episode in schema_candidates]
        self.schema_groups = {}
        for schema, _ in schema_candidates:
            key = schema_suffix(schema)
            if key not in self.schema_groups:
                fps, image_size = schema
                self.schema_groups[key] = {
                    "schema_key": key,
                    "fps": fps,
                    "image_size": image_size,
                    "num_episodes": 0,
                }
            self.schema_groups[key]["num_episodes"] += 1

        if not schema_candidates:
            if self.skip_invalid_episodes:
                self.episodes = []
                self.fps = 0
                self.image_size = (0, 0)
                self.FEATURES = {}
                return
            raise ValueError(f"No schema-compatible Unreal episodes found under {self.raw_dir}")

        selected_schema = self.target_schema or schema_candidates[0][0]
        self.fps, self.image_size = selected_schema
        self.FEATURES = build_features(self.image_size, self.camera_keys)

        compatible = []
        for schema, episode in schema_candidates:
            episode_dir = episode[0]
            if self.keep_all_schemas or schema == selected_schema:
                compatible.append(episode)
                continue
            self._record_exclusion(episode_dir, selected_schema, schema)

        self.episodes = compatible
        if not self.episodes:
            if self.skip_invalid_episodes:
                self.episodes = []
                self.fps = 0
                self.image_size = (0, 0)
                self.FEATURES = {}
                return
            raise ValueError(f"No compatible Unreal episodes found under {self.raw_dir}")

    def _load_episodes(self):
        loaded = []
        episode_dirs = scan_episode_dirs(self.raw_dir)
        total = len(episode_dirs)
        scan_started = time.time()
        last_progress = scan_started
        logging.info("[scan] found %d episode_meta.json files under %s", total, self.raw_dir)

        for index, episode_dir in enumerate(episode_dirs, start=1):
            logging.debug("[scan] scanning episode %d/%d: %s", index, total, episode_dir)
            meta_path = episode_dir / "episode_meta.json"
            frames_path = episode_dir / "frames.jsonl"
            try:
                if not frames_path.exists():
                    raise ValueError("missing frames.jsonl")
                meta = load_json(meta_path)
                if meta.get("status") != "completed":
                    reason = f"episode status is {meta.get('status')!r}, expected 'completed'"
                    self.failed_episodes.append(
                        {
                            "source_episode_path": str(episode_dir),
                            "stage": "episode_status",
                            "error": reason,
                        }
                    )
                    logging.debug("[scan] skipping non-completed episode at %s: %s", episode_dir, reason)
                    continue

                rgb_meta = load_media_meta(episode_dir, "rgb")
                depth_meta = load_media_meta(episode_dir, "depth") if self.export_depth else {}
                meta, metadata_repairs = validate_media_meta(
                    episode_dir,
                    meta,
                    rgb_meta,
                    depth_meta if self.export_depth else None,
                )
                if metadata_repairs:
                    self.repaired_episodes.append(
                        {
                            "source_episode_path": str(episode_dir),
                            "stage": "metadata_recovery",
                            "action": "recovered_legacy_metadata",
                            "repairs": metadata_repairs,
                        }
                    )
                self.warnings.extend(find_undeclared_media(episode_dir, rgb_meta, "rgb"))
                if self.export_depth:
                    self.warnings.extend(find_undeclared_media(episode_dir, depth_meta, "depth"))

                missing = [camera for camera in self.camera_keys if camera not in (meta.get("camera_names") or [])]
                if missing:
                    raise ValueError(f"missing cameras in episode_meta.json: {missing}")

                frames = load_jsonl(frames_path)
                for frame in frames:
                    for camera in self.camera_keys:
                        if f"camera_pose_{camera}" not in frame or f"K_{camera}" not in frame:
                            raise ValueError(f"frame {frame.get('frame_index')} missing camera fields for {camera}")

                task, task_info = load_task_info(episode_dir)
                logging.debug("[scan] validating fixed camera extrinsics for %s", episode_dir)
                body_from_camera = validate_fixed_extrinsics(
                    episode_dir,
                    frames,
                    self.camera_keys,
                    self.translation_tolerance_m,
                    self.rotation_tolerance_deg,
                )
                loaded.append(
                    (
                        episode_dir,
                        meta,
                        frames,
                        task,
                        task_info,
                        body_from_camera,
                        rgb_meta,
                        depth_meta,
                    )
                )
                logging.debug("[scan] accepted episode %s with %d frames", episode_dir, len(frames))
            except Exception as exc:
                self._record_failure(episode_dir, "episode_scan", exc)

            now = time.time()
            if index == total or now - last_progress >= self.log_interval_seconds:
                logging.info(
                    "[scan] progress %d/%d valid=%d skipped=%d metadata_repaired=%d warnings=%d elapsed=%.1fs",
                    index,
                    total,
                    len(loaded),
                    len(self.failed_episodes),
                    len(self.repaired_episodes),
                    len(self.warnings),
                    now - scan_started,
                )
                last_progress = now

        logging.info(
            "[scan] done total=%d valid=%d skipped=%d metadata_repaired=%d warnings=%d elapsed=%.1fs",
            total,
            len(loaded),
            len(self.failed_episodes),
            len(self.repaired_episodes),
            len(self.warnings),
            time.time() - scan_started,
        )
        return loaded

    def _record_failure(self, episode_dir: Path, stage: str, error: Exception):
        failure = {
            "source_episode_path": str(episode_dir),
            "stage": stage,
            "error": str(error),
        }
        self.failed_episodes.append(failure)
        if self.skip_invalid_episodes:
            logging.debug("Skipping invalid episode at %s during %s: %s", episode_dir, stage, error)
            return
        raise error

    def _record_exclusion(
        self,
        episode_dir: Path,
        selected_schema: tuple[int, tuple[int, int]],
        actual_schema: tuple[int, tuple[int, int]],
    ):
        exclusion = {
            "source_episode_path": str(episode_dir),
            "stage": "schema_filter",
            "reason": "other_schema",
            "selected_schema": schema_suffix(selected_schema),
            "actual_schema": schema_suffix(actual_schema),
        }
        self.excluded_episodes.append(exclusion)
        logging.debug(
            "Excluding episode from schema %s because it belongs to %s: %s",
            exclusion["selected_schema"],
            exclusion["actual_schema"],
            episode_dir,
        )

    def _repair_frames_if_needed(self, episode_dir: Path, meta: dict[str, Any], frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaning = meta.get("_cleaning") or {}
        if cleaning and len(frames) == int(cleaning.get("cleaned_frame_count", len(frames))):
            return frames
        expected = int(meta.get("frame_count", len(frames)))
        if len(frames) == expected:
            return frames

        if self.trim_extra_tail_frame and len(frames) == expected + 1:
            last_frame_index = frames[-1].get("frame_index")
            if last_frame_index == expected:
                repair = {
                    "source_episode_path": str(episode_dir),
                    "stage": "frame_count_repair",
                    "action": "trimmed_extra_tail_frame",
                    "meta_frame_count": expected,
                    "original_frame_lines": len(frames),
                    "used_frame_count": expected,
                    "trimmed_frame_index": last_frame_index,
                }
                self.repaired_episodes.append(repair)
                meta["_trimmed_extra_tail_frame"] = True
                logging.debug(
                    "Trimming one extra tail frame in %s: meta.frame_count=%d frames.jsonl=%d",
                    episode_dir,
                    expected,
                    len(frames),
                )
                return frames[:expected]

        raise ValueError(f"frame_count mismatch: meta={meta.get('frame_count')} frames={len(frames)}")

    def _clean_frames_if_needed(
        self,
        episode_dir: Path,
        meta: dict[str, Any],
        frames: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        if not self.clean_invalid_data:
            return frames
        if meta.get("_cleaning"):
            return frames

        fps = float(meta.get("sample_rate_hz") or 10.0)
        decision = build_cleaning_decision(frames, fps)
        decision["source_episode_path"] = str(episode_dir)
        meta["_cleaning"] = decision

        if decision["decision"] in {"drop_direction_invalid", "drop_empty_after_clean", "drop_static_episode"}:
            self.excluded_episodes.append(
                {
                    "source_episode_path": str(episode_dir),
                    "stage": "data_cleaning",
                    "reason": decision["decision"],
                    "drop_reasons": decision.get("drop_reasons", []),
                    "validity_rules": decision.get("validity_rules", []),
                    "original_frame_count": decision.get("original_frame_count"),
                    "cleaned_frame_count": decision.get("cleaned_frame_count"),
                    "stuck_frames": decision.get("stuck_frames"),
                }
            )
            logging.debug(
                "Excluding episode during data cleaning: %s reason=%s",
                episode_dir,
                decision["decision"],
            )
            return None

        if decision["decision"] == "keep_compressed":
            indices = [int(index) for index in decision["source_frame_indices"]]
            cleaned_frames = [frames[index] for index in indices]
            self.repaired_episodes.append(
                {
                    "source_episode_path": str(episode_dir),
                    "stage": "data_cleaning",
                    "action": "compressed_stuck_frames",
                    "original_frame_count": len(frames),
                    "cleaned_frame_count": len(cleaned_frames),
                    "dropped_stuck_frame_count": decision.get("dropped_stuck_frame_count", 0),
                    "compressed_stuck_segments": decision.get("compressed_stuck_segments", []),
                }
            )
            logging.debug(
                "Compressed stuck frames in %s: %d -> %d",
                episode_dir,
                len(frames),
                len(cleaned_frames),
            )
            return cleaned_frames

        return frames

    def for_schema(self, schema: tuple[int, tuple[int, int]]) -> "UnrealEpisodeCollection":
        return UnrealEpisodeCollection(
            raw_dir=self.raw_dir,
            camera_keys=self.camera_keys,
            get_task_idx=self.get_task_idx,
            translation_tolerance_m=self.translation_tolerance_m,
            rotation_tolerance_deg=self.rotation_tolerance_deg,
            skip_invalid_episodes=True,
            target_schema=schema,
            trim_extra_tail_frame=self.trim_extra_tail_frame,
            export_depth=self.export_depth,
            copy_rgb_mp4=self.copy_rgb_mp4,
            clean_invalid_data=self.clean_invalid_data,
            initial_episodes=self.schema_valid_episodes,
            initial_failures=self.failed_episodes,
            initial_repairs=self.repaired_episodes,
            initial_exclusions=[],
            initial_warnings=self.warnings,
            log_interval_seconds=self.log_interval_seconds,
        )

    def for_episodes(
        self,
        episodes: list[tuple],
        *,
        target_schema: tuple[int, tuple[int, int]] | None = None,
    ) -> "UnrealEpisodeCollection":
        source_paths = {str(episode[0]) for episode in episodes}
        return UnrealEpisodeCollection(
            raw_dir=self.raw_dir,
            camera_keys=self.camera_keys,
            get_task_idx=self.get_task_idx,
            translation_tolerance_m=self.translation_tolerance_m,
            rotation_tolerance_deg=self.rotation_tolerance_deg,
            skip_invalid_episodes=True,
            target_schema=target_schema,
            trim_extra_tail_frame=self.trim_extra_tail_frame,
            export_depth=self.export_depth,
            copy_rgb_mp4=self.copy_rgb_mp4,
            clean_invalid_data=self.clean_invalid_data,
            initial_episodes=episodes,
            initial_failures=[],
            initial_repairs=[
                item
                for item in self.repaired_episodes
                if item.get("source_episode_path") in source_paths
            ],
            initial_exclusions=[],
            initial_warnings=[
                item
                for item in self.warnings
                if item.get("source_episode_path") in source_paths
            ],
            log_interval_seconds=self.log_interval_seconds,
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        for episode in self.episodes:
            episode_dir, meta, frames, task, task_info, body_from_camera, rgb_meta, depth_meta = episode
            cleaning = meta.get("_cleaning") or {}
            source_frame_indices = cleaning.get("source_frame_indices") or list(range(len(frames)))
            original_frame_count = int(cleaning.get("original_frame_count") or len(frames))
            if cleaning:
                original_frame_tasks, task_mapping = resolve_frame_tasks(task_info, original_frame_count, task)
                frame_tasks = [original_frame_tasks[int(index)] for index in source_frame_indices]
                task_mapping = {
                    **task_mapping,
                    "cleaning_frame_filter": {
                        "decision": cleaning.get("decision"),
                        "original_frame_count": original_frame_count,
                        "cleaned_frame_count": len(frames),
                    },
                }
            else:
                frame_tasks, task_mapping = resolve_frame_tasks(task_info, len(frames), task)
            task_indices = {
                frame_task: self.get_task_idx(frame_task)
                for frame_task in dict.fromkeys(frame_tasks)
            }
            task_idx = task_indices.get(task, next(iter(task_indices.values())))
            if task_mapping.get("status") == "fallback":
                self.repaired_episodes.append(
                    {
                        "source_episode_path": str(episode_dir),
                        "stage": "task_mapping",
                        "action": "fallback_to_first_non_empty_task",
                        "reason": task_mapping.get("reason", ""),
                        "task": task,
                    }
                )
            try:
                episode = UnrealEpisode(
                    episode_dir,
                    meta,
                    frames,
                    self.camera_keys,
                    task,
                    task_idx,
                    task_info,
                    body_from_camera,
                    rgb_meta,
                    depth_meta,
                    frame_tasks,
                    task_indices,
                    task_mapping,
                    export_depth=self.export_depth,
                    copy_rgb_mp4=self.copy_rgb_mp4,
                    source_frame_indices=[int(index) for index in source_frame_indices],
                    original_frame_count=original_frame_count,
                )
            except Exception as exc:
                self._record_failure(episode_dir, "episode_prepare", exc)
                continue

            self.prepared_episodes.append(
                {
                    "source_episode_path": str(episode_dir),
                    "original_episode_index": int(meta.get("episode_index", -1)),
                    "frame_count": len(frames),
                    "task": task,
                }
            )
            yield episode

    def sync_successful_episodes_from_output(self, root: Path):
        """从实际写出的 extras metadata 回读成功 episode，避免把仅提交到队列的任务误报为成功。"""
        extras_path = root / "meta" / "episodes_extras.jsonl"
        extras = load_jsonl_dicts(extras_path)
        if not extras:
            logging.warning("No episodes_extras.jsonl found or no extras written at %s", extras_path)
            self.successful_episodes = []
        else:
            self.successful_episodes = [
                {
                    "source_episode_path": str(item.get("source_episode_path", "")),
                    "episode_index": item.get("episode_index"),
                    "original_episode_index": item.get("original_episode_index"),
                    "frame_count": item.get("frame_count"),
                    "task": item.get("task", ""),
                    "depth_decode_stats": item.get("depth_decode_stats", {}),
                }
                for item in extras
            ]

        success_sources = {item["source_episode_path"] for item in self.successful_episodes if item["source_episode_path"]}
        failed_sources = {
            item.get("source_episode_path")
            for item in self.failed_episodes
            if item.get("source_episode_path")
        }
        for item in self.prepared_episodes:
            source_path = item["source_episode_path"]
            if source_path not in success_sources and source_path not in failed_sources:
                self.failed_episodes.append(
                    {
                        "source_episode_path": source_path,
                        "stage": "output_validation",
                        "error": "episode was submitted but no episodes_extras entry was written",
                    }
                )
                failed_sources.add(source_path)

    def build_report(self, root: Path, started_at: str, completed_at: str | None, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "raw_dir": str(self.raw_dir),
            "output_root": str(root),
            "camera_keys": self.camera_keys,
            "selected_schema": {
                "fps": self.fps,
                "image_size": self.image_size,
                "schema_key": schema_suffix((self.fps, self.image_size)) if self.fps else "",
            },
            "schema_groups": list(self.schema_groups.values()),
            "clean_invalid_data": self.clean_invalid_data,
            "cleaning_policy": cleaning_policy() if self.clean_invalid_data else None,
            "num_prepared": len(self.prepared_episodes),
            "num_successful": len(self.successful_episodes),
            "num_failed": len(self.failed_episodes),
            "num_repaired": len(self.repaired_episodes),
            "num_excluded": len(self.excluded_episodes),
            "num_warnings": len(self.warnings),
            "prepared_episodes": self.prepared_episodes,
            "successful_episodes": self.successful_episodes,
            "failed_episodes": self.failed_episodes,
            "repaired_episodes": self.repaired_episodes,
            "excluded_episodes": self.excluded_episodes,
            "warnings": self.warnings,
        }


def write_conversion_report(root: Path, report: dict[str, Any]):
    report_path = root / "meta" / "unreal_conversion_report.json"
    write_json(report_path, report)
    logging.info(
        "Wrote conversion report: %s (successful=%s failed=%s)",
        report_path,
        report.get("num_successful"),
        report.get("num_failed"),
    )


def load_completed_scene_report(root: Path, require_depth: bool = True) -> dict[str, Any] | None:
    report_path = root / "meta" / "unreal_conversion_report.json"
    required_paths = [
        report_path,
        root / "meta" / "info.json",
        root / "meta" / "episodes.jsonl",
        root / "meta" / "episodes_extras.jsonl",
        root / "episodes_extras.parquet",
        root / "data",
        root / "videos",
    ]
    if require_depth:
        required_paths.append(root / "images")
    if any(not path.exists() for path in required_paths):
        return None

    try:
        report = load_json(report_path)
    except Exception as exc:
        logging.warning("Could not read existing conversion report at %s: %s", report_path, exc)
        return None

    if report.get("status") != "completed":
        return None
    if int(report.get("num_successful", 0)) <= 0:
        return None
    sidecars = report.get("sidecars") or {}
    depth_report = sidecars.get("depth_sidecars") or {}
    if require_depth and depth_report.get("status") not in (None, "completed"):
        return None
    return report


def has_scene_output(root: Path) -> bool:
    if not root.exists():
        return False
    try:
        next(root.iterdir())
    except StopIteration:
        return False
    return True


def default_scan_cache_path(output_dir: Path) -> Path:
    return output_dir / "unreal_scan_cache.pkl"


def selected_scan_cache_path(output_dir: Path, value: str | None) -> Path:
    return Path(value) if value else default_scan_cache_path(output_dir)


def _normalized_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def save_scan_cache(path: Path, collection: UnrealEpisodeCollection, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SCAN_CACHE_VERSION,
        "created_at": utc_now_iso(),
        "raw_dir": _normalized_path(collection.raw_dir),
        "camera_keys": collection.camera_keys,
        "translation_tolerance_m": collection.translation_tolerance_m,
        "rotation_tolerance_deg": collection.rotation_tolerance_deg,
        "trim_extra_tail_frame": collection.trim_extra_tail_frame,
        "split_by_schema": bool(args.split_by_schema),
        "export_depth": collection.export_depth,
        "copy_rgb_mp4": collection.copy_rgb_mp4,
        "clean_invalid_data": collection.clean_invalid_data,
        "episodes": collection.schema_valid_episodes,
        "failed_episodes": collection.failed_episodes,
        "repaired_episodes": collection.repaired_episodes,
        "excluded_episodes": collection.excluded_episodes,
        "warnings": collection.warnings,
    }
    with path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("Wrote Unreal scan cache: %s (episodes=%d)", path, len(collection.schema_valid_episodes))


def load_scan_cache(path: Path, raw_dir: Path, camera_keys: list[str], args) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Scan cache does not exist: {path}")
    with path.open("rb") as file:
        payload = pickle.load(file)

    if payload.get("version") != SCAN_CACHE_VERSION:
        raise ValueError(f"Unsupported scan cache version: {payload.get('version')}")
    expected_raw_dir = _normalized_path(raw_dir)
    if payload.get("raw_dir") != expected_raw_dir:
        raise ValueError(f"Scan cache raw_dir mismatch: cache={payload.get('raw_dir')} current={expected_raw_dir}")
    if list(payload.get("camera_keys") or []) != camera_keys:
        raise ValueError(f"Scan cache camera_keys mismatch: cache={payload.get('camera_keys')} current={camera_keys}")
    if bool(payload.get("trim_extra_tail_frame")) != bool(args.trim_extra_tail_frame):
        raise ValueError("Scan cache trim_extra_tail_frame mismatch.")
    if bool(payload.get("split_by_schema")) != bool(args.split_by_schema):
        raise ValueError("Scan cache split_by_schema mismatch.")
    if bool(payload.get("clean_invalid_data", False)) != bool(getattr(args, "clean_invalid_data", False)):
        raise ValueError("Scan cache clean_invalid_data mismatch.")
    if not getattr(args, "skip_depth", False) and payload.get("export_depth") is False:
        raise ValueError("Scan cache was created with --skip_depth and cannot be reused for depth export.")
    if float(payload.get("translation_tolerance_m")) != float(args.extrinsic_tolerance_translation_m):
        raise ValueError("Scan cache extrinsic_tolerance_translation_m mismatch.")
    if float(payload.get("rotation_tolerance_deg")) != float(args.extrinsic_tolerance_rotation_deg):
        raise ValueError("Scan cache extrinsic_tolerance_rotation_deg mismatch.")
    logging.info("Loaded Unreal scan cache: %s (episodes=%d)", path, len(payload.get("episodes") or []))
    return payload


def collection_from_scan_cache(
    payload: dict[str, Any],
    raw_dir: Path,
    camera_keys: list[str],
    args,
) -> UnrealEpisodeCollection:
    return UnrealEpisodeCollection(
        raw_dir=raw_dir,
        camera_keys=camera_keys,
        get_task_idx=lambda _task: 0,
        translation_tolerance_m=args.extrinsic_tolerance_translation_m,
        rotation_tolerance_deg=args.extrinsic_tolerance_rotation_deg,
        skip_invalid_episodes=args.skip_invalid_episodes,
        keep_all_schemas=args.split_by_schema,
        trim_extra_tail_frame=args.trim_extra_tail_frame,
        initial_episodes=list(payload.get("episodes") or []),
        initial_failures=list(payload.get("failed_episodes") or []),
        initial_repairs=list(payload.get("repaired_episodes") or []),
        initial_exclusions=list(payload.get("excluded_episodes") or []),
        initial_warnings=list(payload.get("warnings") or []),
        export_depth=not getattr(args, "skip_depth", False),
        copy_rgb_mp4=getattr(args, "copy_rgb_mp4", False),
        log_interval_seconds=getattr(args, "log_interval_seconds", 30.0),
        clean_invalid_data=getattr(args, "clean_invalid_data", False),
    )


def quarantine_incomplete_scene_output(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = root.with_name(f"{root.name}.incomplete.{timestamp}")
    target = base
    suffix = 1
    while target.exists():
        suffix += 1
        target = root.with_name(f"{base.name}.{suffix}")
    root.rename(target)
    return target


def validate_lerobot_dataset(repo_id: str, root: str | Path):
    del repo_id  # Validation is intentionally local and must never query the Hub.
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise ValueError(f"LeRobot info.json is missing: {info_path}")
    info = load_json(info_path)
    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes == 0:
        raise ValueError("Number of episodes is 0.")
    chunks_size = int(info.get("chunks_size", 1000))
    video_keys = [
        key
        for key, feature in (info.get("features") or {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    for episode_index in range(total_episodes):
        chunk = episode_index // chunks_size
        data_path = root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        if not data_path.exists():
            raise ValueError(f"Parquet file is missing: {data_path}")
        for video_key in video_keys:
            video_path = root / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            if not video_path.exists():
                raise ValueError(f"Video file is missing: {video_path}")


def normalize_parquet_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), ensure_ascii=False, default=json_default)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, default=json_default)
    return value


def write_episode_extras_parquet(root: Path) -> dict[str, Any]:
    extras_path = root / "meta" / "episodes_extras.jsonl"
    rows = load_jsonl_dicts(extras_path)
    output_path = root / "episodes_extras.parquet"
    if not rows:
        return {
            "status": "skipped",
            "reason": "missing_or_empty_episodes_extras_jsonl",
            "source": str(extras_path),
            "output": str(output_path),
            "num_rows": 0,
        }

    normalized_rows = [
        {key: normalize_parquet_value(value) for key, value in row.items()}
        for row in rows
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(normalized_rows).to_parquet(output_path, index=False)
    return {
        "status": "completed",
        "source": str(extras_path),
        "output": str(output_path),
        "num_rows": len(normalized_rows),
    }


def summarize_depth_sidecars(root: Path, camera_keys: list[str]) -> dict[str, Any]:
    extras_path = root / "meta" / "episodes_extras.jsonl"
    extras = load_jsonl_dicts(extras_path)
    report: dict[str, Any] = {
        "status": "completed",
        "source": str(extras_path),
        "output_root": str(root / "images"),
        "num_episodes": len(extras),
        "num_depth_files": 0,
        "missing": [],
    }
    if not extras:
        report["status"] = "skipped"
        report["reason"] = "missing_or_empty_episodes_extras_jsonl"
        return report

    for item in extras:
        episode_index = item.get("episode_index")
        if episode_index is None:
            report["missing"].append(
                {
                    "episode_index": episode_index,
                    "reason": "missing_episode_index",
                }
            )
            continue

        chunk = int(episode_index) // 1000
        for camera in camera_keys:
            target_dir = (
                root
                / "images"
                / f"chunk-{chunk:03d}"
                / f"observation.depth.{camera}"
                / f"episode_{int(episode_index):06d}"
            )
            expected_frames = int(item.get("frame_count", 0))
            depth_paths = sorted(target_dir.glob("*.png")) if target_dir.exists() else []
            if len(depth_paths) != expected_frames:
                report["missing"].append(
                    {
                        "episode_index": int(episode_index),
                        "camera": camera,
                        "reason": "depth_frame_count_mismatch",
                        "expected": expected_frames,
                        "actual": len(depth_paths),
                    }
                )
                continue
            report["num_depth_files"] += len(depth_paths)

    if report["missing"]:
        report["status"] = "failed"
    return report


def write_scene_sidecars(root: Path, camera_keys: list[str], export_depth: bool = True) -> dict[str, Any]:
    depth_sidecars = (
        summarize_depth_sidecars(root, camera_keys)
        if export_depth
        else {"status": "skipped", "reason": "depth_export_disabled"}
    )
    return {
        "episodes_extras_parquet": write_episode_extras_parquet(root),
        "depth_sidecars": depth_sidecars,
    }


def group_episodes_by_scene(episodes: list[tuple]) -> dict[str, list[tuple]]:
    groups: dict[str, list[tuple]] = {}
    for episode in episodes:
        episode_dir = episode[0]
        scene_id, _user_id = infer_source_ids(episode_dir)
        groups.setdefault(scene_id or "unknown_scene", []).append(episode)
    return groups


def group_episodes_by_schema(episodes: list[tuple]) -> dict[tuple[int, tuple[int, int]], list[tuple]]:
    groups: dict[tuple[int, tuple[int, int]], list[tuple]] = {}
    for episode in episodes:
        schema = episode_schema(episode[1])
        groups.setdefault(schema, []).append(episode)
    return groups


def repair_stage_counts(repaired_episodes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in repaired_episodes:
        stage = str(item.get("stage") or "unknown")
        counts[stage] = counts.get(stage, 0) + 1
    return counts


def run_conversion(
    collection: UnrealEpisodeCollection,
    root: Path,
    dataset_name: str,
    args,
    started_at: str,
    progress_label: str | None = None,
) -> dict[str, Any]:
    if len(collection) == 0:
        completed_at = utc_now_iso()
        report = collection.build_report(root, started_at, completed_at, "no_valid_episodes")
        write_conversion_report(root, report)
        raise ValueError(f"No compatible Unreal episodes found under {collection.raw_dir}")

    label = progress_label or str(root)
    logging.info(
        "%s prepared episodes=%d output=%s scan_failed=%d",
        label,
        len(collection),
        root,
        len(collection.failed_episodes),
    )
    resolved_pix_fmt = select_video_pixel_format(collection.image_size, codec=args.codec, pix_fmt=args.pix_fmt)
    logging.info("%s media fps=%s image_size=%s pix_fmt=%s", label, collection.fps, collection.image_size, resolved_pix_fmt)

    from utils.lerobot.lerobot_creater import LeRobotCreator

    creator = LeRobotCreator(
        root=str(root),
        robot_type=UnrealEpisodeCollection.ROBOT_TYPE,
        fps=collection.fps,
        features=collection.FEATURES,
        num_workers=max(1, args.num_processes),
        num_video_encoders=max(1, int(max(1, args.num_processes) * 1.75)),
        codec=args.codec,
        pix_fmt=resolved_pix_fmt,
        has_extras=True,
        progress_interval_seconds=args.log_interval_seconds,
    )
    collection.get_task_idx = creator.add_task

    start_time = time.time()
    last_submit_progress = start_time
    status = "failed"
    sidecar_report: dict[str, Any] = {}
    try:
        for episode_index, episode in enumerate(collection, start=1):
            logging.debug("%s submitting episode %s/%s: %s", label, episode_index, len(collection), episode.episode_dir)
            creator.submit_episode(episode)
            now = time.time()
            if episode_index == len(collection) or now - last_submit_progress >= args.log_interval_seconds:
                logging.info(
                    "%s submit progress %d/%d elapsed=%.1fs",
                    label,
                    episode_index,
                    len(collection),
                    now - start_time,
                )
                last_submit_progress = now

        def log_wait_progress(snapshot: dict[str, Any]):
            active = snapshot.get("active") or []
            active_text = ""
            if active:
                parts = []
                for item in active[:3]:
                    source = Path(str(item.get("source_episode_path") or "")).name
                    stage = str(item.get("stage") or "frames")
                    camera = item.get("camera")
                    frames = item.get("frames")
                    total_frames = item.get("total_frames")
                    label_part = f"{source}:{stage}"
                    if camera:
                        label_part += f":{camera}"
                    if total_frames:
                        parts.append(f"{label_part}:{frames}/{total_frames}")
                    else:
                        parts.append(f"{label_part}:{frames}")
                if len(active) > 3:
                    parts.append(f"+{len(active) - 3} active")
                active_text = " active=" + ", ".join(parts)
            logging.info(
                "%s %s progress completed=%d/%d failed=%d elapsed=%.1fs%s",
                label,
                snapshot.get("stage"),
                snapshot.get("completed", 0),
                snapshot.get("total", len(collection)),
                snapshot.get("failed", 0),
                time.time() - start_time,
                active_text,
            )

        logging.info("%s waiting for worker processes and video encoders", label)
        worker_results = creator.wait(
            progress_callback=log_wait_progress,
            log_interval_seconds=args.log_interval_seconds,
        )
        for result in worker_results:
            if result.get("status") == "failed":
                collection.failed_episodes.append(
                    {
                        "source_episode_path": result.get("source_episode_path", ""),
                        "stage": "episode_worker",
                        "error": result.get("error", "unknown worker failure"),
                    }
                )
        logging.info("%s reading written episode metadata from %s", label, root / "meta" / "episodes_extras.jsonl")
        collection.sync_successful_episodes_from_output(root)
        logging.info("%s writing and validating scene sidecars", label)
        sidecar_report = write_scene_sidecars(root, collection.camera_keys, export_depth=collection.export_depth)
        if sidecar_report["depth_sidecars"].get("status") == "failed":
            raise ValueError("Depth sidecar validation failed")
        logging.info("%s validating generated LeRobot dataset at %s", label, root)
        validate_lerobot_dataset(repo_id=dataset_name, root=root)
        status = "completed"
    finally:
        if status != "completed":
            collection.sync_successful_episodes_from_output(root)
            if not sidecar_report:
                sidecar_report = write_scene_sidecars(root, collection.camera_keys, export_depth=collection.export_depth)
        completed_at = utc_now_iso()
        report = collection.build_report(root, started_at, completed_at, status)
        report["sidecars"] = sidecar_report
        write_conversion_report(root, report)

    logging.info(
        "%s done successful=%d failed=%d elapsed=%.1fs output=%s",
        label,
        len(collection.successful_episodes),
        len(collection.failed_episodes),
        time.time() - start_time,
        root,
    )
    report = collection.build_report(root, started_at, utc_now_iso(), status)
    report["sidecars"] = sidecar_report
    return report


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.overwrite_incomplete and not args.resume:
        raise ValueError("--overwrite_incomplete requires --resume.")
    if args.reuse_scan_cache and args.scan_cache is None and not args.resume:
        raise ValueError("--reuse_scan_cache without --scan_cache requires --resume so the default output_dir cache path is well-defined.")

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_keys = parse_camera_keys(args.camera_keys)
    scan_cache_path = selected_scan_cache_path(output_dir, args.scan_cache)
    started_at = utc_now_iso()
    if args.dataset_name:
        logging.warning("--dataset_name is deprecated for Unreal scene-grouped export and will be ignored.")
    logging.info("Starting Unreal conversion raw_dir=%s output_dir=%s cameras=%s", raw_dir, output_dir, camera_keys)

    if args.reuse_scan_cache:
        payload = load_scan_cache(scan_cache_path, raw_dir, camera_keys, args)
        collection = collection_from_scan_cache(payload, raw_dir, camera_keys, args)
    else:
        collection = UnrealEpisodeCollection(
            raw_dir=raw_dir,
            camera_keys=camera_keys,
            get_task_idx=lambda _task: 0,
            translation_tolerance_m=args.extrinsic_tolerance_translation_m,
            rotation_tolerance_deg=args.extrinsic_tolerance_rotation_deg,
            skip_invalid_episodes=args.skip_invalid_episodes,
            keep_all_schemas=args.split_by_schema,
            trim_extra_tail_frame=args.trim_extra_tail_frame,
            export_depth=not args.skip_depth,
            copy_rgb_mp4=args.copy_rgb_mp4,
            log_interval_seconds=args.log_interval_seconds,
            clean_invalid_data=args.clean_invalid_data,
        )
        if args.resume or args.scan_cache:
            save_scan_cache(scan_cache_path, collection, args)

    if not collection.schema_valid_episodes:
        completed_at = utc_now_iso()
        report = collection.build_report(output_dir, started_at, completed_at, "no_valid_episodes")
        write_json(output_dir / "unreal_conversion_report.json", report)
        raise ValueError(f"No schema-compatible Unreal episodes found under {raw_dir}")

    repairs_by_stage = repair_stage_counts(collection.repaired_episodes)
    logging.info(
        "[compat] valid=%d skipped=%d repaired_total=%d repairs_by_stage=%s warnings=%d",
        len(collection.schema_valid_episodes),
        len(collection.failed_episodes),
        len(collection.repaired_episodes),
        repairs_by_stage,
        len(collection.warnings),
    )

    schema_groups = group_episodes_by_schema(collection.schema_valid_episodes) if args.split_by_schema else {None: collection.schema_valid_episodes}
    planned_groups: list[tuple[tuple[int, tuple[int, int]] | None, str, list[tuple]]] = []
    for schema, schema_episodes in sorted(
        schema_groups.items(),
        key=lambda item: schema_suffix(item[0]) if item[0] is not None else "",
    ):
        scene_groups = group_episodes_by_scene(schema_episodes)
        for scene_id, scene_episodes in sorted(scene_groups.items()):
            planned_groups.append((schema, scene_id, scene_episodes))
    logging.info(
        "[plan] schemas=%d scene_groups=%d episodes=%d",
        len(schema_groups),
        len(planned_groups),
        len(collection.schema_valid_episodes),
    )

    group_reports: list[dict[str, Any]] = []
    group_errors: list[dict[str, str]] = []
    for group_index, (schema, scene_id, scene_episodes) in enumerate(planned_groups, start=1):
        schema_key = schema_suffix(schema) if schema is not None else ""
        scene_root = output_dir / schema_key / scene_id if schema_key else output_dir / scene_id
        scene_dataset_name = f"{schema_key}_{scene_id}" if schema_key else scene_id
        scene_label = f"[scene {group_index}/{len(planned_groups)} schema={schema_key or 'default'} scene={scene_id}]"
        if args.resume:
            completed_report = load_completed_scene_report(scene_root, require_depth=not args.skip_depth)
            if completed_report is not None:
                logging.info("%s skipped completed output=%s", scene_label, scene_root)
                completed_report["scene_id"] = scene_id
                completed_report["schema_key"] = schema_key
                completed_report["resume_action"] = "skipped_completed"
                group_reports.append(completed_report)
                continue
            if has_scene_output(scene_root):
                if not args.overwrite_incomplete:
                    raise RuntimeError(
                        f"Incomplete scene output exists at {scene_root}; "
                        "rerun with --resume --overwrite_incomplete to rebuild it."
                    )
                quarantine_root = quarantine_incomplete_scene_output(scene_root)
                logging.warning(
                    "%s moved incomplete output before resume rebuild: %s -> %s",
                    scene_label,
                    scene_root,
                    quarantine_root,
                )

        logging.info("%s start episodes=%d output=%s", scene_label, len(scene_episodes), scene_root)
        scene_collection = collection.for_episodes(scene_episodes, target_schema=schema)
        try:
            report = run_conversion(
                scene_collection,
                scene_root,
                scene_dataset_name,
                args,
                started_at,
                progress_label=scene_label,
            )
            report["scene_id"] = scene_id
            report["schema_key"] = schema_key
            group_reports.append(report)
        except Exception as exc:
            logging.exception("%s failed", scene_label)
            group_errors.append({"scene_id": scene_id, "schema_key": schema_key, "error": str(exc)})

    completed_at = utc_now_iso()
    top_report = {
        "status": "completed" if not group_errors else "completed_with_errors",
        "started_at": started_at,
        "completed_at": completed_at,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "camera_keys": camera_keys,
        "schema_groups": list(collection.schema_groups.values()),
        "group_reports": group_reports,
        "group_errors": group_errors,
        "scan_failures": collection.failed_episodes,
        "repaired_episodes": collection.repaired_episodes,
        "warnings": collection.warnings,
    }
    top_report_path = output_dir / "unreal_conversion_report.json"
    write_json(top_report_path, top_report)
    logging.info("Wrote conversion report: %s", top_report_path)

    if group_errors and not group_reports:
        raise RuntimeError(f"All scene conversions failed; see {top_report_path}")


if __name__ == "__main__":
    main()
