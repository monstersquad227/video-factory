#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频生成与装配模块（流水线 step 3，合并 AI 图生视频 + 逐镜头定长装配）。

两阶段（``--stage ai|assemble|all``，默认 all）：
- 阶段 A（aigen.py，wan3.0-video）：对 ``asset_type=video`` 的镜头，以 ``image/image_NNN.png``
  为首帧生成真实动态片段 ``video/ai_NNN.mp4``（audio=false，duration=clamp(ceil(shot),2,上限)）；
  并发 + 重试 + 断点续跑；单镜头失败不中断，记入 videogen.json，由阶段 B 回退。
- 阶段 B（本文件，ffmpeg）：把**每个** shot 装配成严格等长的 ``video/shot_NNN.mp4``：
  image/motion_graphic/video回退 → 关键帧 Ken Burns（zoompan）；video 且有 ai_NNN →
  scale/crop + fps 归一 + tpad 补尾 + 精确裁剪到 shot.duration；首末镜头淡入淡出。

推荐用 ``--run-dir`` 一站式定位 ``shots.json`` / ``image`` / ``video``：

    python 03-video/generate.py --run-dir ./2026-11-26_loss-aversion
    python 03-video/generate.py --run-dir <run> --stage assemble      # 只装配（跳过 AI）
    python 03-video/generate.py --run-dir <run> --stage ai --dry-run  # 预览 AI payload

也兼容显式路径（--shots/--images/--clips/--output）与旧 Scene 模式（--durations）。
ffmpeg 解析顺序：--ffmpeg > FFMPEG_BIN/IMAGEIO_FFMPEG_EXE > imageio_ffmpeg > 系统 PATH。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import project as project_io  # noqa: E402
from common import prompts as prompt_kit  # noqa: E402


logger = logging.getLogger("video.generate")

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = _SCRIPT_DIR / "output"
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_RESOLUTION = "1080x1920"     # 9:16 竖屏
DEFAULT_FPS = 30
DEFAULT_ZOOM = 1.25                  # Ken Burns 最大缩放（1.25 = 放大到 125%，缓动克制）
DEFAULT_UPSCALE = 2                  # zoompan 前放大倍数（抗抖动）
DEFAULT_CRF = 20
DEFAULT_X264_PRESET = "medium"
DEFAULT_MOTION = "auto"
DEFAULT_FADE = 0.5                   # 首/末镜头淡入淡出秒数（镜头内，不改变时长）
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# 运镜预设（--motion 可覆盖 prompts.resolve_motion_preset 的推断结果）
ALL_PRESETS = [
    "zoom_in", "pan_right", "zoom_out", "pan_left",
    "zoom_in_pan_right", "pan_up", "zoom_out_pan_left", "pan_down", "static",
]


# --------------------------------------------------------------------------- #
# 异常与数据结构
# --------------------------------------------------------------------------- #
class VideoError(Exception):
    """视频生成基础异常。"""


class FFmpegNotFound(VideoError):
    """未找到可用的 ffmpeg 二进制。"""


@dataclass
class ShotClip:
    """一个镜头的素材来源 + 时长 + 动效信息（Shot 模式）。"""

    shot_id: Any
    scene_id: Any
    duration: float
    asset_type: str            # image / video / motion_graphic
    motion: str                # motion_preset
    transition: str
    source: str                # 'ai'（用 AI 片段）/ 'kenburns'（关键帧运镜，含 video 回退）
    keyframe_path: Optional[Path] = None
    clip_path: Optional[Path] = None
    is_first: bool = False
    is_last: bool = False


@dataclass
class ShotVideo:
    """单个镜头的渲染结果。"""

    shot_id: Any
    scene_id: Any
    video_path: Path
    duration: float
    frames: int
    source: str
    motion: str
    asset_type: str
    actual_seconds: Optional[float]
    elapsed_seconds: float
    file_size: int


@dataclass
class BatchResult:
    """整批装配结果。"""

    output_dir: Path
    videos: list
    manifest_path: Path
    resolution: str
    fps: int
    total_elapsed_seconds: float
    ffmpeg_bin: str
    config_snapshot: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# ffmpeg 二进制定位
# --------------------------------------------------------------------------- #
def resolve_ffmpeg(explicit: Optional[str] = None) -> str:
    """按优先级解析 ffmpeg 可执行文件路径，找不到抛 FFmpegNotFound。"""
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return str(p.resolve())
        if shutil.which(explicit):
            return shutil.which(explicit)  # type: ignore[return-value]
        raise FFmpegNotFound(f"指定的 ffmpeg 不存在: {explicit}")

    for var in ("FFMPEG_BIN", "IMAGEIO_FFMPEG_EXE"):
        raw = os.getenv(var)
        if raw and Path(raw).expanduser().exists():
            return str(Path(raw).expanduser().resolve())

    try:
        import imageio_ffmpeg  # 延迟导入：未安装则退回系统 ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:  # noqa: BLE001
        pass

    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff

    raise FFmpegNotFound(
        "未找到 ffmpeg。请任选其一安装：\n"
        "  1) pip install imageio-ffmpeg   （自带静态二进制，本模块会自动探测）\n"
        "  2) brew install ffmpeg          （系统级，step 7 也更方便）\n"
        "或用 --ffmpeg /path/to/ffmpeg 显式指定。"
    )


# --------------------------------------------------------------------------- #
# zoompan 滤镜构造
# --------------------------------------------------------------------------- #
def _motion_expr(motion: str, frames: int, zmax: float) -> tuple[str, str, str]:
    """返回 zoompan 的 (z, x, y) 表达式字符串。

    以输出帧号 on 归一化进度 p = on/(frames-1) ∈ [0,1] 做线性插值，
    与帧数无关，保证不同时长的场景动效速度观感一致。
    """
    denom = max(frames - 1, 1)
    p = f"(on/{denom})"
    cx = "iw/2-(iw/zoom/2)"     # 水平居中
    cy = "ih/2-(ih/zoom/2)"     # 垂直居中
    z_in = f"1.0+({zmax}-1.0)*{p}"      # 1.0 → zmax
    z_out = f"{zmax}-({zmax}-1.0)*{p}"  # zmax → 1.0

    if motion == "zoom_in":
        return z_in, cx, cy
    if motion == "zoom_out":
        return z_out, cx, cy
    if motion == "pan_right":
        return f"{zmax}", f"(iw-iw/zoom)*{p}", cy
    if motion == "pan_left":
        return f"{zmax}", f"(iw-iw/zoom)*(1-{p})", cy
    if motion == "pan_down":
        return f"{zmax}", cx, f"(ih-ih/zoom)*{p}"
    if motion == "pan_up":
        return f"{zmax}", cx, f"(ih-ih/zoom)*(1-{p})"
    if motion == "zoom_in_pan_right":
        return z_in, f"(iw-iw/zoom)*{p}", cy
    if motion == "zoom_out_pan_left":
        return z_out, f"(iw-iw/zoom)*(1-{p})", cy
    # static：极轻微推进，避免完全死板（仍属 Ken Burns 范畴）
    return f"1.0+({min(zmax,1.06)}-1.0)*{p}", cx, cy


def _parse_resolution(value: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d{2,5})\s*[xX*×:]\s*(\d{2,5})\s*", value or "")
    if not m:
        raise VideoError(f"分辨率格式非法: {value!r}，应形如 1080x1920")
    w, h = int(m.group(1)), int(m.group(2))
    if w % 2 or h % 2:
        raise VideoError(f"分辨率宽高需为偶数（libx264 yuv420p 要求），当前 {w}x{h}")
    return w, h


def probe_duration(ffmpeg_bin: str, path: Path) -> Optional[float]:
    """用 ffmpeg/mutagen 读取生成视频的真实时长（秒）。"""
    try:
        from mutagen import File as MutagenFile
        info = MutagenFile(str(path))
        if info is not None and getattr(info, "info", None) is not None:
            return round(float(info.info.length), 3)
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run([ffmpeg_bin, "-hide_banner", "-i", str(path)],
                             capture_output=True, text=True, timeout=30)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", out.stderr or "")
        if m:
            h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            return round(h * 3600 + mn * 60 + s, 3)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


# --------------------------------------------------------------------------- #
# Shot 模式：逐镜头装配（Ken Burns / AI 片段裁剪 / 回退）
# --------------------------------------------------------------------------- #
def _shot_stem(shot_id: Any) -> str:
    """分段视频文件名主干 = 安全化的 shot_id（与 common.project.segment_path 一致，如 s1-1）。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(shot_id)).strip("-._") or "shot"


def load_project_shots(project_dir: str | Path) -> list:
    """读 projects/<id>/script.json，为每个镜头确定素材来源，返回 ShotClip 列表（按时间顺序）。

    - 有 motion 且 video/ai_<shot>.mp4 存在 → source='ai'（用 AI 片段）；
    - 其余（静态/无 AI 片段/回退）→ source='kenburns'（关键帧 + zoompan）；
    - 时长优先 runtime_sec（真实配音），否则名义 duration_sec。
    """
    pdir = Path(project_dir).expanduser().resolve()
    script = project_io.load_script(pdir)
    pairs = list(script.iter_shots())
    total = len(pairs)
    out: list = []
    for i, (scene, shot) in enumerate(pairs):
        duration = float(shot.runtime_sec if shot.runtime_sec is not None else shot.duration_sec)
        if duration <= 0:
            logger.warning("镜头 %s 无有效时长，跳过", shot.id)
            continue
        motion = prompt_kit.resolve_motion_preset(shot)
        if motion not in ALL_PRESETS:
            motion = "zoom_in"
        keyframe = project_io.keyframe_path(pdir, shot.id)
        ai_clip = project_io.ai_clip_path(pdir, shot.id)
        has_motion = bool((shot.motion or "").strip())
        if has_motion and ai_clip.exists() and ai_clip.stat().st_size > 0:
            source, clip = "ai", ai_clip
        else:
            source, clip = "kenburns", None
            if has_motion:
                logger.warning("镜头 %s 无 AI 片段，回退为关键帧运镜（Ken Burns）", shot.id)
        if source == "kenburns" and not keyframe.exists():
            raise VideoError(f"镜头 {shot.id} 找不到关键帧 {keyframe}（请先跑 02-image --project）")
        out.append(ShotClip(
            shot_id=shot.id, scene_id=scene.id, duration=duration,
            asset_type=("video" if has_motion else "image"), motion=motion, transition="",
            source=source, keyframe_path=keyframe if keyframe.exists() else None,
            clip_path=clip, is_first=(i == 0), is_last=(i == total - 1),
        ))
    if not out:
        raise VideoError("script.json 中没有可用镜头")
    return out


def build_shot_vf(*, source: str, motion: str, frames: int, width: int, height: int,
                  fps: int, zoom: float, upscale: int, duration: float,
                  fade: float, fade_in: bool, fade_out: bool) -> str:
    """构造单镜头 -vf 滤镜串（严格产出 duration 时长，不含音轨）。

    - kenburns：关键帧 → 预放大裁剪 → zoompan（复用 _motion_expr）；
    - ai：AI 片段 → scale/crop 到目标分辨率 → fps 归一 → tpad 克隆尾帧兜底
      （保证即使 AI 片段偏短也能凑满 duration）；
    - 首/末镜头叠加 fade（镜头内淡入淡出，不改变时长，保证时间轴同步）。
    """
    parts: list[str] = []
    if source == "ai":
        parts += [
            f"scale={width}:{height}:force_original_aspect_ratio=increase",
            f"crop={width}:{height}",
            f"fps={fps}",
            "setpts=PTS-STARTPTS",
            f"tpad=stop_mode=clone:stop_duration={duration + 1.0:.3f}",
        ]
    else:
        pw, ph = width * upscale, height * upscale
        z, x, y = _motion_expr(motion, frames, zoom)
        parts += [
            f"scale={pw}:{ph}:flags=lanczos:force_original_aspect_ratio=increase",
            f"crop={pw}:{ph}",
            f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={width}x{height}:fps={fps}",
        ]
    fd = min(fade, duration / 2.0) if fade and fade > 0 else 0.0
    if fd > 0 and fade_in:
        parts.append(f"fade=t=in:st=0:d={fd:.3f}")
    if fd > 0 and fade_out:
        parts.append(f"fade=t=out:st={max(duration - fd, 0.0):.3f}:d={fd:.3f}")
    parts.append("format=yuv420p")
    return ",".join(parts)


def build_shot_command(ffmpeg_bin: str, shot: "ShotClip", out_path: Path, *,
                       width: int, height: int, fps: int, zoom: float, upscale: int,
                       crf: int, x264_preset: str, fade: float) -> tuple[list[str], int]:
    """构造单镜头 ffmpeg 命令，返回 (argv, frames)。"""
    frames = max(int(round(shot.duration * fps)), 2)
    vf = build_shot_vf(
        source=shot.source, motion=shot.motion, frames=frames, width=width, height=height,
        fps=fps, zoom=zoom, upscale=upscale, duration=shot.duration, fade=fade,
        fade_in=shot.is_first, fade_out=shot.is_last,
    )
    inp = shot.clip_path if shot.source == "ai" else shot.keyframe_path
    argv = [
        ffmpeg_bin, "-y", "-i", str(inp), "-vf", vf,
        "-frames:v", str(frames), "-r", str(fps),
        "-c:v", "libx264", "-preset", x264_preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
        str(out_path),
    ]
    return argv, frames


def make_shot_video(ffmpeg_bin: str, shot: "ShotClip", out_path: Path, *,
                    width: int, height: int, fps: int, zoom: float, upscale: int,
                    crf: int, x264_preset: str, fade: float, dry_run: bool = False) -> "ShotVideo":
    """渲染单个镜头视频。dry_run=True 时只构造命令不执行。"""
    argv, frames = build_shot_command(
        ffmpeg_bin, shot, out_path, width=width, height=height, fps=fps, zoom=zoom,
        upscale=upscale, crf=crf, x264_preset=x264_preset, fade=fade,
    )
    if dry_run:
        logger.info("[dry-run] 镜头 %s [%s] source=%s motion=%s frames=%d → %s",
                    shot.shot_id, shot.asset_type, shot.source, shot.motion, frames, out_path.name)
        logger.debug("cmd: %s", " ".join(_shell_quote(a) for a in argv))
        return ShotVideo(shot_id=shot.shot_id, scene_id=shot.scene_id, video_path=out_path,
                         duration=shot.duration, frames=frames, source=shot.source,
                         motion=shot.motion, asset_type=shot.asset_type,
                         actual_seconds=None, elapsed_seconds=0.0, file_size=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    logger.info("→ 渲染镜头 %s [%s] source=%s motion=%s dur=%.2fs frames=%d → %s",
                shot.shot_id, shot.asset_type, shot.source, shot.motion,
                shot.duration, frames, out_path.name)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as exc:
        raise VideoError(f"镜头 {shot.shot_id} 渲染超时: {exc}") from exc
    except OSError as exc:
        raise VideoError(f"镜头 {shot.shot_id} 调用 ffmpeg 失败: {exc}") from exc
    elapsed = round(time.perf_counter() - started, 2)
    if proc.returncode != 0 or not out_path.exists():
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        raise VideoError(f"镜头 {shot.shot_id} ffmpeg 渲染失败 (rc={proc.returncode}):\n  "
                         + "\n  ".join(tail))
    actual = probe_duration(ffmpeg_bin, out_path)
    size = out_path.stat().st_size
    logger.info("✓ 镜头 %s: %s | %.2fs | %d 帧 | %.1fs 编码 | 实际 %s",
                shot.shot_id, out_path.name, shot.duration, frames, elapsed,
                f"{actual:.3f}s" if actual is not None else "未知")
    return ShotVideo(shot_id=shot.shot_id, scene_id=shot.scene_id, video_path=out_path,
                     duration=shot.duration, frames=frames, source=shot.source,
                     motion=shot.motion, asset_type=shot.asset_type,
                     actual_seconds=actual, elapsed_seconds=elapsed, file_size=size)


def make_all_shots(shots: list, ffmpeg_bin: str, *, output_dir: Optional[Path] = None,
                   resolution: str = DEFAULT_RESOLUTION, fps: int = DEFAULT_FPS,
                   zoom: float = DEFAULT_ZOOM, upscale: int = DEFAULT_UPSCALE,
                   crf: int = DEFAULT_CRF, x264_preset: str = DEFAULT_X264_PRESET,
                   fade: float = DEFAULT_FADE, dry_run: bool = False) -> BatchResult:
    """批量渲染所有镜头，写出 shot 级 videos.json 清单。"""
    width, height = _parse_resolution(resolution)
    out_dir = Path(output_dir) if output_dir \
        else DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    videos: list = []
    for shot in shots:
        out_path = out_dir / f"{_shot_stem(shot.shot_id)}.mp4"
        videos.append(make_shot_video(
            ffmpeg_bin, shot, out_path, width=width, height=height, fps=fps, zoom=zoom,
            upscale=upscale, crf=crf, x264_preset=x264_preset, fade=fade, dry_run=dry_run,
        ))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "shots",
        "ffmpeg_bin": ffmpeg_bin,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "zoom": zoom,
        "fade": fade,
        "dry_run": dry_run,
        "total_elapsed_seconds": round(time.perf_counter() - started, 2),
        "videos": [
            {
                "shot_id": v.shot_id, "scene_id": v.scene_id, "video_file": v.video_path.name,
                "asset_type": v.asset_type, "source": v.source, "motion": v.motion,
                "target_seconds": v.duration, "actual_seconds": v.actual_seconds,
                "frames": v.frames, "file_size": v.file_size,
            }
            for v in videos
        ],
    }
    manifest_path = out_dir / "videos.json"
    if not dry_run:
        try:
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        except OSError as exc:
            raise VideoError(f"写入 videos.json 失败: {exc}") from exc

    return BatchResult(
        output_dir=out_dir, videos=videos, manifest_path=manifest_path,
        resolution=f"{width}x{height}", fps=fps,
        total_elapsed_seconds=manifest["total_elapsed_seconds"], ffmpeg_bin=ffmpeg_bin,
        config_snapshot={"zoom": zoom, "upscale": upscale, "crf": crf,
                         "x264_preset": x264_preset, "fade": fade},
    )


def _run_ai_stage(args: argparse.Namespace, ffmpeg_bin: str, project_dir: Path) -> None:
    """阶段 A：对有 motion 的镜头调用 wan3.0-video 生成 AI 片段 video/ai_<shot>.mp4。"""
    try:
        import aigen
    except ImportError as exc:  # pragma: no cover
        raise VideoError(f"无法导入 aigen（AI 视频生成模块）: {exc}") from exc
    overrides = {
        "model": args.ai_model, "resolution": args.ai_resolution, "ratio": args.ai_ratio,
        "max_duration": args.ai_max_duration, "concurrency": args.ai_concurrency,
    }
    logger.info("=== 阶段 A：wan3.0-video 图生视频（有 motion 的镜头）===")
    result = aigen.generate_ai_clips(
        project_dir, env_path=Path(args.env).expanduser() if args.env else None,
        shot_ids=args.shot, resume=not args.no_resume_ai, dry_run=args.dry_run,
        ffmpeg_bin=ffmpeg_bin, skip_motion_less=not args.include_static, **overrides)
    logger.info("AI 片段：成功 %d / 跳过 %d / 失败 %d → %s",
                result.ok_count, result.skipped_count, result.failed_count, result.output_dir)
    if result.failed_count:
        logger.warning("%d 个 AI 镜头失败，装配阶段将回退为关键帧运镜", result.failed_count)


def _run_project(args: argparse.Namespace, ffmpeg_bin: str, project_dir: Path) -> int:
    """阶段 B：逐镜头装配为定长 video/<shot>.mp4，并回写 script.json 的 status/outputs.video。"""
    shots = load_project_shots(project_dir)
    if args.motion in ALL_PRESETS:
        for s in shots:
            s.motion = args.motion
    out_dir = project_io.video_dir(project_dir)
    ai_n = sum(1 for s in shots if s.source == "ai")
    logger.info("=== 阶段 B：逐镜头定长装配 ===")
    logger.info("ffmpeg: %s", ffmpeg_bin)
    logger.info("待装配镜头: %d（ai=%d, kenburns=%d）→ %s", len(shots), ai_n, len(shots) - ai_n, out_dir)

    result = make_all_shots(
        shots, ffmpeg_bin, output_dir=out_dir,
        resolution=args.resolution, fps=args.fps, zoom=args.zoom, upscale=args.upscale,
        crf=args.crf, x264_preset=args.x264_preset, fade=args.fade, dry_run=args.dry_run,
    )

    if not args.dry_run:
        script = project_io.load_script(project_dir)
        for v in result.videos:
            project_io.set_shot_status(script, str(v.shot_id), video="done")
            project_io.set_shot_output(script, str(v.shot_id),
                                       video=project_io.rel(project_dir, v.video_path))
        project_io.save_script(script, project_dir / "script.json")

    logger.info("— 结果 —")
    for v in result.videos:
        actual = f"{v.actual_seconds:.3f}s" if v.actual_seconds is not None else "-"
        logger.info("  镜头 %-6s sc=%-4s %-8s motion=%-18s 目标 %.2fs 实际 %-9s %s",
                    v.shot_id, v.scene_id, v.source, v.motion, v.duration, actual, v.video_path.name)
    logger.info("  分辨率    : %s @ %dfps | 总耗时 %.2fs", result.resolution, result.fps,
                result.total_elapsed_seconds)
    print(json.dumps({
        "ok": True, "mode": "project", "dry_run": args.dry_run,
        "output_dir": str(result.output_dir.resolve()),
        "manifest": str(result.manifest_path.resolve()) if not args.dry_run else None,
        "resolution": result.resolution, "fps": result.fps, "shot_count": len(result.videos),
        "videos": [{"shot_id": v.shot_id, "scene_id": v.scene_id, "source": v.source,
                    "video_path": str(v.video_path.resolve()), "motion": v.motion,
                    "target_seconds": v.duration, "actual_seconds": v.actual_seconds}
                   for v in result.videos],
    }, ensure_ascii=False))
    return 0


def _shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'" if re.search(r"[^\w@%+=:,./-]", text) else text


def _human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{int(num)} B" if unit == "B" else f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} TB"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video.generate",
        description="视频生成与装配：wan3.0-video 图生视频 + 逐镜头定长分段（Ken Burns / AI 片段裁剪）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project", required=True, help="短剧项目目录（projects/<id>）：自动定位 script.json / keyframes / video")
    parser.add_argument("--stage", choices=["ai", "assemble", "all"], default="all",
                        help="ai=只生成AI片段, assemble=只装配定长段, all=两者（默认）")
    parser.add_argument("--shot", action="append", help="仅对指定镜头生成 AI 片段（可重复或逗号分隔）")
    parser.add_argument("--include-static", action="store_true", help="motion 为空的镜头也生成 AI 视频")
    parser.add_argument("--no-resume-ai", action="store_true", help="AI 阶段不跳过已存在的 ai_<shot>.mp4")
    parser.add_argument("--ai-model", help="wan3.0-video 模型名（覆盖 .env VIDEO_MODEL）")
    parser.add_argument("--ai-resolution", choices=["480P", "720P", "1080P"], help="AI 视频分辨率档位")
    parser.add_argument("--ai-ratio", help="AI 视频宽高比，如 9:16")
    parser.add_argument("--ai-max-duration", type=int, help="AI 单镜头生成时长上限秒")
    parser.add_argument("--ai-concurrency", type=int, help="AI 并发任务数")
    parser.add_argument("--env", help="指定 .env 路径（AI 阶段）")
    parser.add_argument("--output", help=f"输出目录（默认 {DEFAULT_OUTPUT_ROOT}/<时间戳>）")
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION, help="输出分辨率 WxH（宽高需偶数）")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="帧率")
    parser.add_argument("--zoom", type=float, default=DEFAULT_ZOOM, help="Ken Burns 最大缩放倍数")
    parser.add_argument("--upscale", type=int, default=DEFAULT_UPSCALE, help="zoompan 前预放大倍数（抗抖动）")
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF, help="libx264 CRF 质量（越小越好）")
    parser.add_argument("--preset", dest="x264_preset", default=DEFAULT_X264_PRESET,
                        help="libx264 编码预设（ultrafast..veryslow）")
    parser.add_argument("--motion", default=DEFAULT_MOTION,
                        help=f"运镜预设覆盖：auto（默认，按 motion/action 推断）/ {' / '.join(ALL_PRESETS)}")
    parser.add_argument("--fade", type=float, default=DEFAULT_FADE,
                        help="Shot 模式首/末镜头淡入淡出秒数（0 关闭）")
    parser.add_argument("--ffmpeg", help="显式指定 ffmpeg 可执行文件路径")
    parser.add_argument("--dry-run", action="store_true", help="只构造并打印命令，不实际渲染")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    try:
        if args.motion not in (["auto"] + ALL_PRESETS):
            raise VideoError(f"--motion 非法: {args.motion}；可选 auto/{'/'.join(ALL_PRESETS)}")
        if args.zoom <= 1.0:
            raise VideoError(f"--zoom 需 > 1.0，当前 {args.zoom}")
        if args.upscale < 1:
            raise VideoError(f"--upscale 需 >= 1，当前 {args.upscale}")

        ffmpeg_bin = resolve_ffmpeg(args.ffmpeg)
        pdir = Path(args.project).expanduser().resolve()
        if not (pdir / "script.json").exists():
            raise VideoError(f"项目缺少 script.json: {pdir}（请先跑 00-scripts 与 01-voice）")
        if args.stage in ("ai", "all"):
            _run_ai_stage(args, ffmpeg_bin, pdir)
        if args.stage in ("assemble", "all"):
            return _run_project(args, ffmpeg_bin, pdir)
        return 0

    except FFmpegNotFound as exc:
        logger.error("%s", exc)
        print(json.dumps({"ok": False, "error_type": "FFmpegNotFound", "error": str(exc)}, ensure_ascii=False))
        return 5
    except VideoError as exc:
        logger.error("视频生成失败: %s", exc)
        print(json.dumps({"ok": False, "error_type": "VideoError", "error": str(exc)}, ensure_ascii=False))
        return 1
    except KeyboardInterrupt:
        logger.error("用户中断")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("未预期错误: %s", exc)
        print(json.dumps({"ok": False, "error_type": exc.__class__.__name__, "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
