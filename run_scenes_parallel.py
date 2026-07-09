#!/usr/bin/env python3
"""按 scene 并行运行 unreal.py 转换。

unreal.py 的输出本来就按 scene 分目录（output_dir/<scene_id>/...），且 --raw_dir
支持直接指向单个 scene 目录。因此可以为每个 scene 起一个独立的 unreal.py 进程、
共用同一个 output_dir，彼此写各自的 scene 子目录、互不冲突，无需事后合并。

这样可把「所有 scene 串行求和」的总时长压缩到「最慢的单个 scene」，充分利用多核。

示例：
    python run_scenes_parallel.py \
        --raw_dir /home/szt/simu/StuckSaved \
        --output_dir /home/szt/simu/xnav-stuck-parallel \
        --max_parallel 9 \
        -- --camera_keys front,rear,left,right --skip_invalid_episodes \
           --trim_extra_tail_frame --clean_invalid_data --skip_depth --num_processes 8

`--` 之后的所有参数会原样透传给每个 unreal.py 实例（不要在其中再写 --raw_dir/--output_dir）。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 限制每个 worker 进程里数值库的线程池规模。多个 scene 实例 × 多个 worker 并发时，
# 若每个进程都按机器总核数（如 256）开 BLAS/OpenMP 线程池，会造成线程超额订阅、
# 调度颠簸。把它们限制为小值即可，因为单个 worker 本就串行处理帧。
THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def discover_scene_dirs(raw_dir: Path) -> list[Path]:
    """返回 raw_dir 下形如 scene_* 的子目录（按名称排序）。"""
    return sorted(p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("scene_"))


def run_one_scene(
    scene_dir: Path,
    output_dir: Path,
    log_dir: Path,
    python_exe: str,
    unreal_script: Path,
    passthrough: list[str],
    blas_threads: int,
) -> tuple[str, int, float, Path]:
    """转换单个 scene，返回 (scene_id, returncode, elapsed_sec, log_path)。"""
    scene_id = scene_dir.name
    log_path = log_dir / f"{scene_id}.log"
    cmd = [
        python_exe,
        str(unreal_script),
        "--raw_dir",
        str(scene_dir),
        "--output_dir",
        str(output_dir),
        *passthrough,
    ]
    env = os.environ.copy()
    for name in THREAD_LIMIT_ENV_VARS:
        env[name] = str(blas_threads)
    # OpenCV 解码线程上限（unreal.py 读取该变量限制 cv2 / ffmpeg 解码线程）。
    env["XNAV_DECODE_THREADS"] = str(blas_threads)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# CMD: {' '.join(cmd)}\n")
        log_file.write(f"# ENV: {' '.join(f'{n}={blas_threads}' for n in THREAD_LIMIT_ENV_VARS)}\n\n")
        log_file.flush()
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    return scene_id, proc.returncode, time.time() - start, log_path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run unreal.py conversion in parallel, one process per scene.",
    )
    parser.add_argument("--raw_dir", required=True, help="包含 scene_* 子目录的根目录。")
    parser.add_argument("--output_dir", required=True, help="所有 scene 实例共用的输出目录。")
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=0,
        help="最大并发 scene 数（0 = 全部 scene 同时跑）。",
    )
    parser.add_argument(
        "--scenes",
        default="",
        help="只跑这些 scene（逗号分隔，如 scene_0016,scene_0019）；留空跑全部。",
    )
    parser.add_argument(
        "--blas_threads",
        type=int,
        default=2,
        help="每个实例的 BLAS/OpenMP 线程上限，防止线程超额订阅。",
    )
    parser.add_argument(
        "--python_exe",
        default=sys.executable,
        help="运行 unreal.py 的 Python 解释器（默认当前解释器）。",
    )
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = parse_args()
    # argparse 会把分隔用的 '--' 留在 passthrough 首位，去掉它。
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    raw_dir = Path(args.raw_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    unreal_script = Path(__file__).resolve().parent / "unreal.py"
    log_dir = output_dir / "_parallel_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = discover_scene_dirs(raw_dir)
    if args.scenes:
        wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
        scene_dirs = [p for p in scene_dirs if p.name in wanted]
    if not scene_dirs:
        print(f"[parallel] 在 {raw_dir} 下没有找到 scene_* 目录", file=sys.stderr)
        return 1

    max_parallel = args.max_parallel or len(scene_dirs)
    print(f"[parallel] scenes={len(scene_dirs)} max_parallel={max_parallel} output={output_dir}")
    for p in scene_dirs:
        print(f"  - {p.name}")

    overall_start = time.time()
    results: list[tuple[str, int, float, Path]] = []
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(
                run_one_scene,
                scene_dir,
                output_dir,
                log_dir,
                args.python_exe,
                unreal_script,
                passthrough,
                args.blas_threads,
            ): scene_dir.name
            for scene_dir in scene_dirs
        }
        for future in as_completed(futures):
            scene_id, code, elapsed, log_path = future.result()
            status = "OK" if code == 0 else f"FAILED(code={code})"
            print(f"[parallel] {scene_id:14s} {status:18s} elapsed={elapsed:7.1f}s log={log_path}")
            results.append((scene_id, code, elapsed, log_path))

    total_elapsed = time.time() - overall_start
    failures = [r for r in results if r[1] != 0]
    print("\n[parallel] ===== summary =====")
    print(f"[parallel] total wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"[parallel] scenes ok={len(results) - len(failures)} failed={len(failures)}")
    if results:
        slowest = max(results, key=lambda r: r[2])
        print(f"[parallel] slowest scene: {slowest[0]} ({slowest[2]:.1f}s)")
    for scene_id, code, _elapsed, log_path in failures:
        print(f"[parallel]   FAILED {scene_id}: see {log_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
