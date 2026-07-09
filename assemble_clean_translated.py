#!/usr/bin/env python3
"""Assemble a cleaned LeRobot dataset from translated + compressed replacements.

Policy: from the already-converted full dataset `translated`, keep every episode EXCEPT:
  - drop_static_episode  -> excluded
  - drop_direction_invalid -> excluded
  - keep_compressed      -> excluded here, replaced by the compressed version
                            produced in `xnav-stuck-final`

Unaffected buckets (no static, no compressed) are hardlinked wholesale.
Affected buckets are rebuilt: surviving translated episodes + final compressed
episodes, re-indexed contiguously, with rebuilt meta (episodes/stats/tasks/info).

Per-episode stats only cover feature columns (no index/episode_index/task_index),
so they are reused verbatim; only the bookkeeping columns inside each parquet
(episode_index, index, task_index) and file names are rewritten.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TRANS = "/mnt/datasets/translated"
FINAL = "/home/szt/simu/xnav-stuck-final"
OUT = "/mnt/datasets/clean_translated"
DIAG = "/tmp/full_diag.pkl"
SRC_MAP = "/tmp/translated_src_map.pkl"
CHUNK_SIZE = 1000

# 严阈值策略:方向无效集也丢弃(连同全程静止集);keep_compressed 由 final 压缩版替换。
EXCLUDE_FROM_TRANS = {"drop_static_episode", "drop_direction_invalid", "keep_compressed"}


def relkey(p: str) -> str:
    return "/".join(p.rstrip("/").split("/")[-3:])


def load_meta_lines(path: str) -> dict[int, dict]:
    """episodes.jsonl / episodes_stats.jsonl -> {episode_index: record}."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            out[d["episode_index"]] = d
    return out


def load_tasks(path: str) -> dict[int, str]:
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            out[d["task_index"]] = d["task"]
    return out


def video_cams(bucket_dir: str) -> list[str]:
    vd = Path(bucket_dir) / "videos" / "chunk-000"
    if not vd.exists():
        return []
    return sorted(p.name for p in vd.iterdir() if p.is_dir())


def hardlink_tree(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # cp -al: recursive hardlink (instant, shares blocks)
    try:
        subprocess.run(["cp", "-al", src, dst], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"hardlink failed {src} -> {dst}") from exc
    staging = Path(dst) / ".lerobot_staging"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)


def rewrite_episode(src_bucket, old_idx, new_idx, src_tasks, unified_tasks, dst_bucket, cams):
    """Copy one episode into dst with rewritten episode_index/index/task_index."""
    src_pq = f"{src_bucket}/data/chunk-000/episode_{old_idx:06d}.parquet"
    t = pq.read_table(src_pq)
    n = t.num_rows

    def set_col(table, name, values):
        i = table.schema.get_field_index(name)
        typ = table.schema.field(name).type
        return table.set_column(i, name, pa.array(values, type=typ))

    t = set_col(t, "episode_index", [new_idx] * n)
    # 注意:translated/final 的 `index` 列约定是「每集 0 基(== frame_index)」,
    # 不是全局累加。保留源 index 不动,保持与硬链接桶一致。
    old_ti = t.column("task_index").to_pylist()
    new_ti = [unified_tasks[src_tasks[oti]] for oti in old_ti]
    t = set_col(t, "task_index", new_ti)

    dst_pq = f"{dst_bucket}/data/chunk-000/episode_{new_idx:06d}.parquet"
    os.makedirs(os.path.dirname(dst_pq), exist_ok=True)
    pq.write_table(t, dst_pq)

    for cam in cams:
        src_v = f"{src_bucket}/videos/chunk-000/{cam}/episode_{old_idx:06d}.mp4"
        dst_v = f"{dst_bucket}/videos/chunk-000/{cam}/episode_{new_idx:06d}.mp4"
        os.makedirs(os.path.dirname(dst_v), exist_ok=True)
        try:
            os.link(src_v, dst_v)  # 同盘:瞬时硬链
        except OSError:
            shutil.copy2(src_v, dst_v)  # 跨盘(如 final 在本地、输出在 /mnt):拷贝

    return n


def build_affected_bucket(schema, scene, trans_eps, final_eps, dst_bucket):
    """trans_eps: list of (old_idx) kept from translated (sorted).
    final_eps: list of (old_idx) from final (sorted)."""
    src_trans = f"{TRANS}/{schema}/{scene}"
    src_final = f"{FINAL}/{schema}/{scene}"
    cams = video_cams(src_trans)

    # source meta
    trans_ep_meta = load_meta_lines(f"{src_trans}/meta/episodes.jsonl")
    trans_stats = load_meta_lines(f"{src_trans}/meta/episodes_stats.jsonl")
    trans_tasks = load_tasks(f"{src_trans}/meta/tasks.jsonl")
    final_ep_meta = load_meta_lines(f"{src_final}/meta/episodes.jsonl") if final_eps else {}
    final_stats = load_meta_lines(f"{src_final}/meta/episodes_stats.jsonl") if final_eps else {}
    final_tasks = load_tasks(f"{src_final}/meta/tasks.jsonl") if final_eps else {}

    # final ordered episode list: (source_tag, old_idx)
    ordered = [("T", oi) for oi in trans_eps] + [("F", oi) for oi in final_eps]

    # unified tasks: union of task strings used by ordered episodes
    used_tasks = []
    for tag, oi in ordered:
        meta = trans_ep_meta if tag == "T" else final_ep_meta
        for ts in meta[oi]["tasks"]:
            if ts not in used_tasks:
                used_tasks.append(ts)
    unified_tasks = {ts: i for i, ts in enumerate(used_tasks)}

    os.makedirs(f"{dst_bucket}/meta", exist_ok=True)
    episodes_jsonl = []
    episodes_stats_jsonl = []
    total_frames = 0
    for new_idx, (tag, old_idx) in enumerate(ordered):
        src_bucket = src_trans if tag == "T" else src_final
        src_tasks = trans_tasks if tag == "T" else final_tasks
        ep_meta = (trans_ep_meta if tag == "T" else final_ep_meta)[old_idx]
        stats = (trans_stats if tag == "T" else final_stats)[old_idx]
        n = rewrite_episode(src_bucket, old_idx, new_idx, src_tasks, unified_tasks, dst_bucket, cams)
        total_frames += n
        episodes_jsonl.append({"episode_index": new_idx, "tasks": ep_meta["tasks"], "length": n})
        new_stats = dict(stats)
        new_stats["episode_index"] = new_idx
        episodes_stats_jsonl.append(new_stats)

    n_ep = len(ordered)
    # write meta
    with open(f"{dst_bucket}/meta/episodes.jsonl", "w", encoding="utf-8") as f:
        for r in episodes_jsonl:
            f.write(json.dumps(r) + "\n")
    with open(f"{dst_bucket}/meta/episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for r in episodes_stats_jsonl:
            f.write(json.dumps(r) + "\n")
    with open(f"{dst_bucket}/meta/tasks.jsonl", "w", encoding="utf-8") as f:
        for ts, ti in unified_tasks.items():
            f.write(json.dumps({"task_index": ti, "task": ts}) + "\n")

    with open(f"{src_trans}/meta/info.json", encoding="utf-8") as f:
        info = json.load(f)
    info["total_episodes"] = n_ep
    info["total_frames"] = total_frames
    info["total_videos"] = n_ep * len(cams)
    info["total_tasks"] = len(unified_tasks)
    info["total_chunks"] = (n_ep - 1) // CHUNK_SIZE + 1 if n_ep else 0
    info["splits"] = {"train": f"0:{n_ep}"}
    with open(f"{dst_bucket}/meta/info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    return n_ep, total_frames


def main():
    global TRANS, FINAL, OUT

    ap = argparse.ArgumentParser()
    ap.add_argument("--translated", default=TRANS, help="源 translated 数据集目录")
    ap.add_argument("--final", default=FINAL, help="压缩后替换 episode 的数据集目录")
    ap.add_argument("--output", default=OUT, help="输出 clean_translated 目录")
    ap.add_argument("--diag", default=DIAG, help="full_diag.pkl 路径")
    ap.add_argument("--src_map", default=SRC_MAP, help="translated_src_map.pkl 路径")
    ap.add_argument("--only", default="", help="只处理这些桶 schema/scene(逗号分隔),用于测试")
    ap.add_argument("--reset", action="store_true", help="先删除输出目录")
    args = ap.parse_args()

    TRANS = args.translated
    FINAL = args.final
    OUT = args.output

    with open(args.diag, "rb") as f:
        diag = pickle.load(f)
    with open(args.src_map, "rb") as f:
        src_map = pickle.load(f)

    # translated buckets -> list of (old_idx, src_ep, decision)
    trans_buckets = collections.defaultdict(list)
    for src_ep, (schema, scene, eidx) in src_map.items():
        trans_buckets[(schema, scene)].append((eidx, src_ep, diag[src_ep][0]))

    # final buckets -> list of (final_old_idx)
    final_buckets = collections.defaultdict(list)
    for f in glob.glob(f"{FINAL}/*/*/meta/episodes_extras.jsonl"):
        parts = f.split(f"{FINAL}/")[1].split("/")
        schema, scene = parts[0], parts[1]
        for line in open(f, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                final_buckets[(schema, scene)].append(d["episode_index"])

    only = set()
    if args.only:
        for x in args.only.split(","):
            x = x.strip()
            if x:
                only.add(tuple(x.split("/")))

    if args.reset and os.path.exists(OUT):
        shutil.rmtree(OUT)

    summary = []
    for (schema, scene), eps in sorted(trans_buckets.items()):
        if only and (schema, scene) not in only:
            continue
        dst_bucket = f"{OUT}/{schema}/{scene}"
        if os.path.exists(dst_bucket):
            shutil.rmtree(dst_bucket)
        n_static = sum(1 for _, _, d in eps if d == "drop_static_episode")
        n_dir = sum(1 for _, _, d in eps if d == "drop_direction_invalid")
        n_comp = sum(1 for _, _, d in eps if d == "keep_compressed")
        final_eps = sorted(final_buckets.get((schema, scene), []))
        affected = (n_static or n_dir or n_comp or final_eps)
        if not affected:
            hardlink_tree(f"{TRANS}/{schema}/{scene}", dst_bucket)
            n_ep = len(eps)
            summary.append((schema, scene, n_ep, "hardlink"))
            print(f"[hardlink] {schema}/{scene}: {n_ep} eps")
        else:
            trans_keep = sorted(oi for oi, _, d in eps if d not in EXCLUDE_FROM_TRANS)
            n_ep, n_fr = build_affected_bucket(schema, scene, trans_keep, final_eps, dst_bucket)
            summary.append((schema, scene, n_ep, f"surgery(-{n_static}st,-{n_dir}dir,+{len(final_eps)}comp)"))
            print(f"[surgery]  {schema}/{scene}: {n_ep} eps ({n_fr} fr)  -{n_static}static -{n_dir}dir +{len(final_eps)}comp")

    total = sum(s[2] for s in summary)
    print(f"\n=== 合计输出 {total} 集, {len(summary)} 桶 ===")


if __name__ == "__main__":
    main()
