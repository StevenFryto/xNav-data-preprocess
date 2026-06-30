# xNav Data Preprocess

## 环境配置
本仓库使用[`uv`](https://docs.astral.sh/uv/getting-started)管理环境，安装好uv后使用如下命令配置环境

```bash
uv sync --all-groups
```

该命令会将环境装在项目目录下的`.venv`中，之后使用`uv run xxx.py`即可用该环境跑某个python程序。

## 概述
本仓库包含xNav对一些开源数据集的格式转换，详见[`docs`](./docs/)。

- [`rgb_pose_to_lerobot.py`](rgb_pose_to_lerobot.py): 给第一视角web video封装了一份代码，详见[`rgb_pose_example/README.md`](examples/rgb_pose_example/README.md)。
- [`lerobot_creator_example.py`](lerobot_creator_example.py): 教程示例代码。
- [`unreal.py`](unreal.py): 将 `3d-simu-ue` 录制出的 raw episode 转为按 scene 组织的 LeRobot v2.1 数据集；使用说明见脚本顶部注释。

## Unreal Saved 转换

当前 UE 录制格式使用四路 RGB MP4 和 HueMp4 深度视频。转换器会将 RGB 写入
LeRobot v2.1 视频 feature，并把 HueMp4 恢复为 `uint16` 毫米深度 PNG sidecar。
四路 RGB、四路 Depth 和轨迹帧数必须完全一致。

```bash
uv run unreal.py \
    --raw_dir /mnt/datasets/Saved \
    --output_dir /path/to/empty-output \
    --camera_keys front,rear,left,right \
    --skip_invalid_episodes \
    --split_by_schema \
    --trim_extra_tail_frame \
    --resume \
    --skip_depth \
    --copy_rgb_mp4
```

- `--skip_invalid_episodes` 用于跳过 front-only 或损坏 episode，并写入转换报告。
- `--split_by_schema` 用于按 FPS 和分辨率拆分混合数据。
- `--trim_extra_tail_frame` 仅修复已知的单条连续尾帧问题，不修改原始数据。
- `--skip_depth` 用于跳过 depth HueMp4 校验、解码和 `images/observation.depth.*` sidecar 导出，只保留 RGB 视频与状态数据。
- `--copy_rgb_mp4` 用于直接复制帧数兼容的源 RGB MP4，避免 RGB 解码和重编码；需要裁掉额外尾帧的 episode 会自动回退到普通重编码路径。
- `--resume` 用于跳过已完整转换的 schema/scene；半成品目录默认报错，可加 `--overwrite_incomplete` 移走半成品并重跑该 scene。
- `--resume` 会在扫描后写出 `output_dir/unreal_scan_cache.pkl`；后续重启可加 `--reuse_scan_cache` 复用扫描结果，跳过逐 episode 外参校验。
- `action` 当前仍复制 `observation.state`，是兼容现有训练配置的占位字段。
- 首次转换建议使用空输出目录；断点续跑只在 schema/scene 粒度生效，不支持单 episode 级续写。

