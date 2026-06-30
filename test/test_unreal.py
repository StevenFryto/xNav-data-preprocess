from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

try:
    import pyarrow  # noqa: F401

    HAS_PARQUET_ENGINE = True
except ImportError:
    HAS_PARQUET_ENGINE = False

from unreal import (
    ACTION_KEY,
    STATE_KEY,
    TASK_DESCRIPTION_KEY,
    UnrealEpisode,
    UnrealEpisodeCollection,
    body_from_camera_for_frame,
    build_features,
    decode_hue_depth_rgb,
    group_episodes_by_scene,
    group_episodes_by_schema,
    has_scene_output,
    find_undeclared_media,
    intrinsic_matrix,
    intrinsic_4,
    collection_from_scan_cache,
    load_completed_scene_report,
    load_scan_cache,
    load_media_meta,
    resolve_frame_tasks,
    save_scan_cache,
    scan_episode_dirs,
    validate_media_meta,
    validate_lerobot_dataset,
    validate_fixed_extrinsics,
    write_episode_extras_parquet,
)


def make_frame(frame_index, body_x_cm, camera_x_cm):
    return {
        "episode_index": 0,
        "frame_index": frame_index,
        "timestamp_wall_sec": float(frame_index),
        "timestamp_sim_sec": float(frame_index),
        "pose": [body_x_cm, 0.0, 0.0, 0.0, 0.0, 0.0],
        "view_mode": "first_person",
        "camera_pose_front": [camera_x_cm, 0.0, 0.0, 0.0, 0.0, 0.0],
        "K_front": [100.0, 0.0, 2.0, 0.0, 110.0, 3.0, 0.0, 0.0, 1.0],
    }


def write_episode(
    root: Path,
    frames: list[dict],
    scene_id: str = "scene_0001",
    user_id: str = "user_0001",
    episode_id: str = "episode_000000",
    width: int = 4,
    height: int = 3,
    fps: int = 10,
    meta_frame_count: int | None = None,
    write_depth: bool = False,
):
    episode_dir = root / scene_id / user_id / episode_id
    (episode_dir / "rgb" / "front").mkdir(parents=True)
    if write_depth:
        (episode_dir / "depth" / "front").mkdir(parents=True)
    else:
        (episode_dir / "depth").mkdir(parents=True)

    meta = {
        "status": "completed",
        "episode_index": 0,
        "scene_id": scene_id,
        "map_name": "Entry",
        "capture_width": width,
        "capture_height": height,
        "sample_rate_hz": fps,
        "frame_count": len(frames) if meta_frame_count is None else meta_frame_count,
        "camera_names": ["front"],
    }
    (episode_dir / "episode_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    rgb_meta = {
        "capture_width": width,
        "capture_height": height,
        "frame_rate_hz": fps,
        "camera_names": ["front"],
        "storage": "png_sequence",
        "encode_video_directly": False,
    }
    depth_meta = {
        "capture_width": width,
        "capture_height": height,
        "frame_rate_hz": fps,
        "camera_names": ["front"],
        "storage": "png_sequence",
        "video_encoding": "PngSequence",
        "depth_unit": "millimeter",
    }
    (episode_dir / "rgb" / "meta.json").write_text(json.dumps(rgb_meta), encoding="utf-8")
    (episode_dir / "depth" / "meta.json").write_text(json.dumps(depth_meta), encoding="utf-8")
    with (episode_dir / "frames.jsonl").open("w", encoding="utf-8") as file:
        for frame in frames:
            file.write(json.dumps(frame) + "\n")

    for index in range(len(frames)):
        image = np.full((height, width, 3), index + 1, dtype=np.uint8)
        Image.fromarray(image).save(episode_dir / "rgb" / "front" / f"{index:05d}.png")
        if write_depth:
            depth = np.full((height, width), index + 1, dtype=np.uint16)
            Image.fromarray(depth).save(episode_dir / "depth" / "front" / f"{index:05d}.png")

    return episode_dir, meta


class UnrealConversionTests(unittest.TestCase):
    def test_load_completed_scene_report_accepts_complete_output(self):
        with tempfile.TemporaryDirectory(prefix="unreal_resume_") as tmp:
            root = Path(tmp)
            meta_dir = root / "meta"
            meta_dir.mkdir(parents=True)
            for directory in ["data", "videos", "images"]:
                (root / directory).mkdir()
            (meta_dir / "info.json").write_text("{}", encoding="utf-8")
            (meta_dir / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
            (meta_dir / "episodes_extras.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "episodes_extras.parquet").touch()
            report = {
                "status": "completed",
                "num_successful": 2,
                "sidecars": {"depth_sidecars": {"status": "completed"}},
            }
            (meta_dir / "unreal_conversion_report.json").write_text(json.dumps(report), encoding="utf-8")

            loaded = load_completed_scene_report(root)

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["num_successful"], 2)

    def test_load_completed_scene_report_rejects_incomplete_output(self):
        with tempfile.TemporaryDirectory(prefix="unreal_resume_") as tmp:
            root = Path(tmp)
            (root / "meta").mkdir(parents=True)
            (root / "meta" / "unreal_conversion_report.json").write_text(
                json.dumps({"status": "failed", "num_successful": 1}),
                encoding="utf-8",
            )

            self.assertIsNone(load_completed_scene_report(root))
            self.assertTrue(has_scene_output(root))

    def test_load_completed_scene_report_accepts_no_depth_output_when_not_required(self):
        with tempfile.TemporaryDirectory(prefix="unreal_resume_") as tmp:
            root = Path(tmp)
            meta_dir = root / "meta"
            meta_dir.mkdir(parents=True)
            for directory in ["data", "videos"]:
                (root / directory).mkdir()
            (meta_dir / "info.json").write_text("{}", encoding="utf-8")
            (meta_dir / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
            (meta_dir / "episodes_extras.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "episodes_extras.parquet").touch()
            report = {
                "status": "completed",
                "num_successful": 1,
                "sidecars": {"depth_sidecars": {"status": "skipped", "reason": "depth_export_disabled"}},
            }
            (meta_dir / "unreal_conversion_report.json").write_text(json.dumps(report), encoding="utf-8")

            self.assertIsNotNone(load_completed_scene_report(root, require_depth=False))
            self.assertIsNone(load_completed_scene_report(root, require_depth=True))

    def test_has_scene_output_ignores_missing_and_empty_dirs(self):
        with tempfile.TemporaryDirectory(prefix="unreal_resume_") as tmp:
            root = Path(tmp)
            missing = root / "missing"
            empty = root / "empty"
            empty.mkdir()

            self.assertFalse(has_scene_output(missing))
            self.assertFalse(has_scene_output(empty))

    def test_scan_cache_round_trips_collection_without_rescan(self):
        with tempfile.TemporaryDirectory(prefix="unreal_scan_cache_") as tmp:
            root = Path(tmp)
            write_episode(root, [make_frame(0, 0.0, 100.0)])
            args = SimpleNamespace(
                split_by_schema=True,
                trim_extra_tail_frame=False,
                extrinsic_tolerance_translation_m=1e-4,
                extrinsic_tolerance_rotation_deg=0.1,
                skip_invalid_episodes=True,
            )
            collection = UnrealEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
                keep_all_schemas=True,
            )
            cache_path = root / "scan_cache.pkl"

            save_scan_cache(cache_path, collection, args)
            payload = load_scan_cache(cache_path, root, ["front"], args)
            cached = collection_from_scan_cache(payload, root, ["front"], args)

            self.assertEqual(len(cached.schema_valid_episodes), 1)
            self.assertEqual(cached.schema_groups["fps10_3x4"]["num_episodes"], 1)

    def test_collection_skip_depth_does_not_require_depth_metadata(self):
        with tempfile.TemporaryDirectory(prefix="unreal_skip_depth_") as tmp:
            root = Path(tmp)
            write_episode(root, [make_frame(0, 0.0, 100.0)])
            shutil.rmtree(next(root.glob("scene_*/user_*/episode_*")) / "depth")

            collection = UnrealEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
                export_depth=False,
            )

            self.assertEqual(len(collection.schema_valid_episodes), 1)

    def test_unreal_episode_copy_rgb_mp4_skips_frame_images(self):
        with tempfile.TemporaryDirectory(prefix="unreal_copy_rgb_") as tmp:
            root = Path(tmp)
            frames = [make_frame(0, 0.0, 100.0)]
            episode_dir, meta = write_episode(root, frames)
            body_from_camera = validate_fixed_extrinsics(episode_dir, frames, ["front"], 1e-4, 0.1)
            source_video = episode_dir / "rgb" / "front.mp4"
            source_video.touch()
            episode = UnrealEpisode(
                episode_dir,
                meta,
                frames,
                ["front"],
                "",
                0,
                [],
                body_from_camera,
                load_media_meta(episode_dir, "rgb"),
                {},
                copy_rgb_mp4=True,
            )
            episode.image_sources = {
                "front": SimpleNamespace(video_path=source_video),
            }

            item, _task = next(iter(episode))

            self.assertEqual(episode.direct_video_paths, {"video.front": str(source_video)})
            self.assertNotIn("video.front", item)
            self.assertIn(STATE_KEY, item)

    def test_scan_cache_rejects_mismatched_camera_keys(self):
        with tempfile.TemporaryDirectory(prefix="unreal_scan_cache_") as tmp:
            root = Path(tmp)
            write_episode(root, [make_frame(0, 0.0, 100.0)])
            args = SimpleNamespace(
                split_by_schema=True,
                trim_extra_tail_frame=False,
                extrinsic_tolerance_translation_m=1e-4,
                extrinsic_tolerance_rotation_deg=0.1,
                skip_invalid_episodes=True,
            )
            collection = UnrealEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
                keep_all_schemas=True,
            )
            cache_path = root / "scan_cache.pkl"

            save_scan_cache(cache_path, collection, args)

            with self.assertRaisesRegex(ValueError, "camera_keys mismatch"):
                load_scan_cache(cache_path, root, ["rear"], args)

    def test_validate_lerobot_dataset_uses_local_metadata_only(self):
        with tempfile.TemporaryDirectory(prefix="unreal_lerobot_") as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            (root / "data" / "chunk-000").mkdir(parents=True)
            (root / "videos" / "chunk-000" / "video.front").mkdir(parents=True)
            info = {
                "total_episodes": 1,
                "chunks_size": 1000,
                "features": {"video.front": {"dtype": "video"}},
            }
            (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
            (root / "data" / "chunk-000" / "episode_000000.parquet").touch()
            (root / "videos" / "chunk-000" / "video.front" / "episode_000000.mp4").touch()

            validate_lerobot_dataset("local-test", root)

    def test_find_undeclared_media_reports_png_residue(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            episode_dir, _ = write_episode(root, [make_frame(0, 0.0, 100.0)])
            rgb_meta = load_media_meta(episode_dir, "rgb")
            rgb_meta["storage"] = "mp4"

            warnings = find_undeclared_media(episode_dir, rgb_meta, "rgb")

            self.assertEqual(len(warnings), 1)
            self.assertEqual(warnings[0]["warning"], "ignored_undeclared_png_files")
            self.assertEqual(warnings[0]["count"], 1)

    def test_resolve_frame_tasks_maps_multiple_segments(self):
        tasks, status = resolve_frame_tasks(
            [
                {"start_frame": 0, "end_frame": 2, "name": "first"},
                {"start_frame": 2, "end_frame": 4, "name": "second"},
            ],
            frame_count=5,
            fallback_task="first",
        )

        self.assertEqual(tasks, ["first", "first", "second", "second", "second"])
        self.assertEqual(status["status"], "mapped")

    def test_resolve_frame_tasks_keeps_blank_segment_name(self):
        tasks, status = resolve_frame_tasks(
            [
                {"start_frame": 0, "end_frame": 1, "name": "first"},
                {"start_frame": 1, "end_frame": 2, "name": ""},
            ],
            frame_count=3,
            fallback_task="first",
        )

        self.assertEqual(tasks, ["first", "", ""])
        self.assertEqual(status["status"], "mapped")

    def test_resolve_frame_tasks_falls_back_on_invalid_bounds(self):
        tasks, status = resolve_frame_tasks(
            [{"start_frame": 0, "end_frame": 99, "name": "first"}],
            frame_count=3,
            fallback_task="first",
        )

        self.assertEqual(tasks, ["first", "first", "first"])
        self.assertEqual(status["status"], "fallback")

    def test_decode_hue_depth_rgb_recovers_six_hue_sectors(self):
        rgb = np.array(
            [
                [
                    [255, 0, 0],
                    [255, 255, 0],
                    [0, 255, 0],
                    [0, 255, 255],
                    [0, 0, 255],
                    [255, 0, 255],
                ]
            ],
            dtype=np.uint8,
        )

        depth, valid = decode_hue_depth_rgb(rgb, 0.0, 20.0)

        np.testing.assert_array_equal(depth[0], np.array([0, 4000, 8000, 12000, 16000, 20000]))
        self.assertTrue(valid.all())

    def test_decode_hue_depth_rgb_marks_dark_and_gray_pixels_invalid(self):
        rgb = np.array([[[0, 0, 0], [15, 0, 0], [100, 100, 100], [250, 12, 4]]], dtype=np.uint8)

        depth, valid = decode_hue_depth_rgb(rgb, 0.0, 20.0)

        np.testing.assert_array_equal(valid[0], np.array([False, False, False, True]))
        np.testing.assert_array_equal(depth[0, :3], np.zeros(3, dtype=np.uint16))
        self.assertLess(int(depth[0, 3]), 1000)

    def test_decode_hue_depth_rgb_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            decode_hue_depth_rgb(np.zeros((2, 2), dtype=np.uint8), 0.0, 20.0)
        with self.assertRaisesRegex(ValueError, "Invalid hue depth range"):
            decode_hue_depth_rgb(np.zeros((2, 2, 3), dtype=np.uint8), 20.0, 20.0)

    def test_validate_media_meta_recovers_legacy_fields(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            episode_dir, meta = write_episode(root, [make_frame(0, 0.0, 100.0)])
            meta.pop("scene_id")
            meta["camera_names"] = []

            recovered, repairs = validate_media_meta(
                episode_dir,
                meta,
                load_media_meta(episode_dir, "rgb"),
                load_media_meta(episode_dir, "depth"),
            )

            self.assertEqual(recovered["scene_id"], "scene_0001")
            self.assertEqual(recovered["camera_names"], ["front"])
            self.assertEqual({item["field"] for item in repairs}, {"scene_id", "camera_names"})

    def test_validate_media_meta_rejects_camera_conflict(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            episode_dir, meta = write_episode(root, [make_frame(0, 0.0, 100.0)])
            depth_meta = load_media_meta(episode_dir, "depth")
            depth_meta["camera_names"] = ["rear"]

            with self.assertRaisesRegex(ValueError, "RGB/Depth camera_names mismatch"):
                validate_media_meta(
                    episode_dir,
                    meta,
                    load_media_meta(episode_dir, "rgb"),
                    depth_meta,
                )

    def test_scan_episode_dirs_accepts_root_and_episode_dir(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            episode_dir, _ = write_episode(root, [make_frame(0, 0.0, 100.0)])

            self.assertEqual(scan_episode_dirs(root), [episode_dir])
            self.assertEqual(scan_episode_dirs(episode_dir), [episode_dir])

    def test_intrinsic_4_extracts_fx_fy_cx_cy(self):
        frame = make_frame(0, 0.0, 100.0)
        self.assertEqual(intrinsic_4(frame, "front"), [100.0, 110.0, 2.0, 3.0])
        self.assertEqual(intrinsic_matrix(frame, "front"), [[100.0, 0.0, 2.0], [0.0, 110.0, 3.0], [0.0, 0.0, 1.0]])

    def test_body_from_camera_is_fixed_when_body_and_camera_move_together(self):
        frames = [
            make_frame(0, 0.0, 100.0),
            make_frame(1, 100.0, 200.0),
        ]
        baseline = validate_fixed_extrinsics(Path("episode_000000"), frames, ["front"], 1e-4, 0.1)

        expected = body_from_camera_for_frame(frames[0], "front")
        np.testing.assert_allclose(baseline["front"], expected, atol=1e-6)

    def test_body_from_camera_change_fails(self):
        frames = [
            make_frame(0, 0.0, 100.0),
            make_frame(1, 100.0, 201.0),
        ]
        with self.assertRaisesRegex(ValueError, "Dynamic body_from_camera"):
            validate_fixed_extrinsics(Path("episode_000000"), frames, ["front"], 1e-4, 0.1)

    def test_unreal_episode_outputs_local_7d_state(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            frames = [
                make_frame(0, 100.0, 200.0),
                make_frame(1, 200.0, 300.0),
            ]
            episode_dir, meta = write_episode(root, frames)
            body_from_camera = validate_fixed_extrinsics(episode_dir, frames, ["front"], 1e-4, 0.1)
            episode = UnrealEpisode(
                episode_dir=episode_dir,
                meta=meta,
                frames=frames,
                camera_keys=["front"],
                task="",
                task_idx=0,
                task_info=[],
                body_from_camera=body_from_camera,
            )

            emitted = [frame for frame, _task in episode]
            self.assertEqual(emitted[0][TASK_DESCRIPTION_KEY].tolist(), [0])
            self.assertEqual(emitted[0]["video.front"].shape, (3, 4, 3))
            np.testing.assert_allclose(emitted[0][STATE_KEY], np.array([0, 0, 0, 0, 0, 0, 1]), atol=1e-6)
            np.testing.assert_allclose(emitted[1][STATE_KEY][:3], np.array([1, 0, 0]), atol=1e-6)
            np.testing.assert_allclose(emitted[1][ACTION_KEY], emitted[1][STATE_KEY], atol=1e-6)

            metadata = episode.metadata
            self.assertEqual(metadata["K_front"], [[100.0, 0.0, 2.0], [0.0, 110.0, 3.0], [0.0, 0.0, 1.0]])
            self.assertIn("Extrinsic_front", metadata)
            self.assertEqual(metadata["fps"], 10)
            self.assertEqual(metadata["capture_width"], 4)
            self.assertEqual(metadata["capture_height"], 3)

    def test_build_features_adds_all_requested_camera_keys(self):
        features = build_features((3, 4), ["front", "rear"])
        self.assertEqual(features["video.front"]["shape"], (3, 4, 3))
        self.assertEqual(features["video.rear"]["shape"], (3, 4, 3))
        self.assertEqual(features[STATE_KEY]["shape"], (7,))

    def test_collection_can_skip_invalid_episode(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            write_episode(root, [make_frame(0, 0.0, 100.0)])

            invalid_dir = root / "scene_0001" / "user_0002" / "episode_000000"
            invalid_dir.mkdir(parents=True)
            (invalid_dir / "episode_meta.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "episode_index": 0,
                        "capture_width": 4,
                        "capture_height": 3,
                        "sample_rate_hz": 10,
                        "frame_count": 1,
                        "camera_names": [],
                    }
                ),
                encoding="utf-8",
            )
            (invalid_dir / "rgb").mkdir()
            (invalid_dir / "depth").mkdir()
            (invalid_dir / "rgb" / "meta.json").write_text(
                json.dumps(
                    {
                        "capture_width": 4,
                        "capture_height": 3,
                        "frame_rate_hz": 10,
                        "camera_names": ["front"],
                        "storage": "png_sequence",
                    }
                ),
                encoding="utf-8",
            )
            (invalid_dir / "depth" / "meta.json").write_text(
                json.dumps(
                    {
                        "capture_width": 4,
                        "capture_height": 3,
                        "frame_rate_hz": 10,
                        "camera_names": ["rear"],
                        "storage": "png_sequence",
                    }
                ),
                encoding="utf-8",
            )
            (invalid_dir / "frames.jsonl").write_text(json.dumps(make_frame(0, 0.0, 100.0)) + "\n", encoding="utf-8")

            collection = UnrealEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
            )

            self.assertEqual(len(collection), 1)
            self.assertEqual(len(collection.failed_episodes), 1)
            self.assertIn("RGB/Depth camera_names mismatch", collection.failed_episodes[0]["error"])

    def test_collection_can_split_mixed_schemas(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            write_episode(root, [make_frame(0, 0.0, 100.0)], user_id="user_0001", fps=10, width=4, height=3)
            write_episode(
                root,
                [make_frame(0, 0.0, 100.0)],
                user_id="user_0002",
                episode_id="episode_000001",
                fps=30,
                width=8,
                height=6,
            )

            collection = UnrealEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
                keep_all_schemas=True,
            )
            self.assertEqual(set(collection.schema_groups), {"fps10_3x4", "fps30_6x8"})

            split_collection = collection.for_schema((30, (6, 8)))
            self.assertEqual(len(split_collection), 1)
            self.assertEqual(split_collection.fps, 30)
            self.assertEqual(split_collection.image_size, (6, 8))
            self.assertEqual(len(split_collection.failed_episodes), 0)
            self.assertEqual(len(split_collection.excluded_episodes), 1)
            self.assertEqual(split_collection.excluded_episodes[0]["reason"], "other_schema")

    def test_group_episodes_by_scene_and_schema(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            ep_a, _ = write_episode(root, [make_frame(0, 0.0, 100.0)], scene_id="scene_a", fps=10, width=4, height=3)
            ep_b, _ = write_episode(root, [make_frame(0, 0.0, 100.0)], scene_id="scene_b", fps=30, width=8, height=6)

            collection = UnrealEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
                keep_all_schemas=True,
            )

            by_scene = group_episodes_by_scene(collection.schema_valid_episodes)
            self.assertEqual(set(by_scene), {"scene_a", "scene_b"})
            self.assertEqual(by_scene["scene_a"][0][0], ep_a)
            self.assertEqual(by_scene["scene_b"][0][0], ep_b)

            by_schema = group_episodes_by_schema(collection.schema_valid_episodes)
            self.assertEqual(set(by_schema), {(10, (3, 4)), (30, (6, 8))})

    def test_for_episodes_keeps_matching_repairs_and_warnings(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            episode_dir, _ = write_episode(root, [make_frame(0, 0.0, 100.0)])
            collection = UnrealEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
            )
            collection.repaired_episodes.append(
                {"source_episode_path": str(episode_dir), "action": "repair"}
            )
            collection.warnings.append(
                {"source_episode_path": str(episode_dir), "warning": "warning"}
            )

            grouped = collection.for_episodes(collection.episodes)

            self.assertEqual(grouped.repaired_episodes[-1]["action"], "repair")
            self.assertEqual(grouped.warnings[-1]["warning"], "warning")

    def test_collection_can_trim_one_extra_tail_frame(self):
        with tempfile.TemporaryDirectory(prefix="unreal_episode_") as tmp:
            root = Path(tmp)
            frames = [make_frame(0, 0.0, 100.0), make_frame(1, 100.0, 200.0)]
            write_episode(root, frames, meta_frame_count=1)

            collection = UnrealEpisodeCollection(
                raw_dir=root,
                camera_keys=["front"],
                get_task_idx=lambda _task: 0,
                translation_tolerance_m=1e-4,
                rotation_tolerance_deg=0.1,
                skip_invalid_episodes=True,
                trim_extra_tail_frame=True,
            )

            self.assertEqual(len(collection), 1)
            self.assertEqual(len(collection.episodes[0][2]), 1)
            self.assertTrue(collection.episodes[0][1]["_trimmed_extra_tail_frame"])
            self.assertEqual(len(collection.repaired_episodes), 1)
            self.assertEqual(collection.repaired_episodes[0]["action"], "trimmed_extra_tail_frame")

    def test_write_episode_extras_parquet(self):
        if not HAS_PARQUET_ENGINE:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory(prefix="unreal_sidecar_") as tmp:
            root = Path(tmp)
            meta_dir = root / "meta"
            meta_dir.mkdir()
            extras = {
                "episode_index": 0,
                "scene_id": "scene_0001",
                "K_front": [[100.0, 0.0, 2.0], [0.0, 110.0, 3.0], [0.0, 0.0, 1.0]],
                "Extrinsic_front": np.eye(4).tolist(),
                "task_info": [{"name": "go"}],
            }
            (meta_dir / "episodes_extras.jsonl").write_text(json.dumps(extras) + "\n", encoding="utf-8")

            report = write_episode_extras_parquet(root)

            self.assertEqual(report["status"], "completed")
            output_path = root / "episodes_extras.parquet"
            self.assertTrue(output_path.exists())
            df = pd.read_parquet(output_path)
            self.assertEqual(df.loc[0, "episode_index"], 0)
            self.assertEqual(df.loc[0, "scene_id"], "scene_0001")
            self.assertIn("100.0", df.loc[0, "K_front"])

    def test_prepare_and_commit_hue_depth_sidecars(self):
        with tempfile.TemporaryDirectory(prefix="unreal_sidecar_") as tmp:
            root = Path(tmp)
            raw_root = root / "raw"
            frames = [make_frame(0, 0.0, 100.0), make_frame(1, 100.0, 200.0)]
            episode_dir, meta = write_episode(raw_root, frames)
            depth_meta = load_media_meta(episode_dir, "depth")
            depth_meta.update(
                {
                    "storage": "mp4",
                    "video_encoding": "HueMp4",
                    "hue_min_meters": 0.0,
                    "hue_max_meters": 20.0,
                }
            )
            (episode_dir / "depth" / "meta.json").write_text(json.dumps(depth_meta), encoding="utf-8")
            (episode_dir / "depth" / "front.mp4").touch()
            body_from_camera = validate_fixed_extrinsics(episode_dir, frames, ["front"], 1e-4, 0.1)
            episode = UnrealEpisode(
                episode_dir,
                meta,
                frames,
                ["front"],
                "",
                0,
                [],
                body_from_camera,
                load_media_meta(episode_dir, "rgb"),
                depth_meta,
            )

            encoded_rgb = np.array([[[255, 255, 0]]], dtype=np.uint8)
            encoded_bgr = encoded_rgb[..., ::-1]

            class FakeCapture:
                def __init__(self, _path):
                    self.frames = [encoded_bgr.copy(), encoded_bgr.copy()]

                def isOpened(self):
                    return True

                def read(self):
                    if not self.frames:
                        return False, None
                    return True, self.frames.pop(0)

                def release(self):
                    pass

            fake_cv2 = SimpleNamespace(VideoCapture=FakeCapture)
            with patch.dict(sys.modules, {"cv2": fake_cv2}):
                prepared = episode.prepare_episode(root / "output")
            episode.commit_prepared_episode(root / "output", 3, prepared)

            depth_dir = root / "output" / "images" / "chunk-000" / "observation.depth.front" / "episode_000003"
            self.assertTrue((depth_dir / "00000.png").exists())
            self.assertTrue((depth_dir / "00001.png").exists())
            with Image.open(depth_dir / "00000.png") as image:
                self.assertEqual(int(np.asarray(image)[0, 0]), 4000)
            self.assertEqual(episode.depth_decode_stats["front"]["frame_count"], 2)


if __name__ == "__main__":
    unittest.main()
