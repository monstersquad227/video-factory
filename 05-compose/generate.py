#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成模块（流水线 step 5）：拼接视频 + 混音（配音 + BGM）+ 烧录字幕 → final.mp4。

把前面几步的产物合成成片：
- 视频：``03-video`` 逐镜头 ``shot_<id>.mp4``（Shot 模式，段数=镜头数）或逐场景 ``scene_<id>.mp4``（Scene 模式）；
- 配音：``01-voice`` 逐场景 ``scene_<id>.mp3``，按场景顺序 concat（统一 48kHz 立体声）；视频段与配音段数量可不同，总时长一致即可对齐；
- 字幕：``04-subtitle`` 的 ``subtitle.srt``，自转 ASS（PlayRes=视频分辨率）烧录，竖屏像素级字号；
- BGM：可选 ``--bgm``，自动循环铺满、压低音量，可选侧链闪避与结尾淡出。

推荐用 ``--run-dir`` 一站式定位 shots.json / audio / video / subtitle.srt，成片写回 run 目录：

    python 05-compose/generate.py --run-dir ./2026-11-26_loss-aversion
    python 05-compose/generate.py --run-dir <run> --bgm assets/bgm.mp3 --duck

也兼容显式路径（--shots/--durations/--videos/--srt）与旧 Scene 模式（仅 --durations）。
"""

from __future__ import annotations

import argparse
import json
import logging
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


logger = logging.getLogger("compose.generate")

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = _SCRIPT_DIR / "output"
_PROJECT_ROOT = _SCRIPT_DIR.parent

# 混音默认值
DEFAULT_NARRATION_VOLUME = 1.0
DEFAULT_BGM_VOLUME = 0.18          # BGM 相对旁白压低，保证人声清晰
DEFAULT_BGM_FADE = 1.5             # BGM 结尾淡出秒数
AUDIO_SAMPLE_RATE = 48000

# 字幕默认样式（竖屏 9:16 优化）：底部居中、白字黑边。配合 original_size 使用像素语义。
DEFAULT_FONT = "Hiragino Sans GB"  # macOS 自带中文字体；可用 --font 覆盖
DEFAULT_FONT_SIZE = 60             # 像素（original_size=视频分辨率时 1:1）
DEFAULT_MARGIN_V = 150             # 底部边距（像素）
DEFAULT_MARGIN_X = 48              # 左右安全边距（像素），避免贴边
DEFAULT_OUTLINE = 2
PRIMARY_COLOUR = "&H00FFFFFF"      # 白色（&HAABBGGRR）
OUTLINE_COLOUR = "&H80000000"      # 半透明黑描边


# --------------------------------------------------------------------------- #
# 异常与数据结构
# --------------------------------------------------------------------------- #
class ComposeError(Exception):
    """合成基础异常。"""


class FFmpegNotFound(ComposeError):
    """未找到可用的 ffmpeg 二进制。"""


@dataclass
class ComposeResult:
    """合成结果。"""

    output_path: Path
    duration_seconds: Optional[float]
    file_size: int
    scene_count: int
    resolution: str
    has_bgm: bool
    has_subtitle: bool
    elapsed_seconds: float
    ffmpeg_bin: str
    command: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# ffmpeg 二进制定位
# --------------------------------------------------------------------------- #
def resolve_ffmpeg(explicit: Optional[str] = None) -> str:
    """按优先级解析 ffmpeg 可执行文件路径。"""
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return str(p.resolve())
        found = shutil.which(explicit)
        if found:
            return found
        raise FFmpegNotFound(f"指定的 ffmpeg 不存在: {explicit}")

    for var in ("FFMPEG_BIN", "IMAGEIO_FFMPEG_EXE"):
        raw = os.getenv(var)
        if raw and Path(raw).expanduser().exists():
            return str(Path(raw).expanduser().resolve())

    try:
        import imageio_ffmpeg
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
        "  1) pip install imageio-ffmpeg\n"
        "  2) brew install ffmpeg\n"
        "或用 --ffmpeg /path/to/ffmpeg 显式指定。"
    )


# --------------------------------------------------------------------------- #
# concat 列表文件
# --------------------------------------------------------------------------- #
def _concat_escape(path: Path) -> str:
    return str(path).replace("'", "'\\''")


def write_concat_list(paths: list[Path], list_file: Path) -> Path:
    """写 concat demuxer 列表文件。"""
    lines = [f"file '{_concat_escape(p)}'" for p in paths]
    list_file.parent.mkdir(parents=True, exist_ok=True)
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_file


# --------------------------------------------------------------------------- #
# 字幕：SRT → ASS（自定 PlayRes，保证像素级字号与正确折行）
# --------------------------------------------------------------------------- #
_SRT_BLOCK_RE = re.compile(
    r"\d+\s*\n(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*\n(.*?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)


def _srt_time_to_seconds(ts: str) -> float:
    m = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})", ts.strip())
    if not m:
        raise ComposeError(f"SRT 时间戳非法: {ts!r}")
    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    ms = int(m.group(4).ljust(3, "0"))
    return h * 3600 + mi * 60 + s + ms / 1000.0


def _format_ass_time(seconds: float) -> str:
    """秒 → ASS 时间戳 H:MM:SS.cc（厘秒）。"""
    if seconds < 0:
        seconds = 0.0
    cs = int(round(seconds * 100))
    h, rem = divmod(cs, 360000)
    mi, rem = divmod(rem, 6000)
    s, cc = divmod(rem, 100)
    return f"{h}:{mi:02d}:{s:02d}.{cc:02d}"


def parse_srt(srt_path: Path) -> list:
    """解析 SRT，返回 [(start_s, end_s, text)]。"""
    text = Path(srt_path).read_text(encoding="utf-8")
    cues = []
    for m in _SRT_BLOCK_RE.finditer(text):
        start = _srt_time_to_seconds(m.group(1))
        end = _srt_time_to_seconds(m.group(2))
        body = m.group(3).strip().replace("\r", "")
        if not body or end <= start:
            continue
        cues.append((start, end, body))
    if not cues:
        raise ComposeError(f"SRT 中没有可用字幕: {srt_path}")
    return cues


def build_ass_style_line(font: str = DEFAULT_FONT, font_size: int = DEFAULT_FONT_SIZE,
                         margin_v: int = DEFAULT_MARGIN_V, margin_x: int = DEFAULT_MARGIN_X,
                         outline: float = DEFAULT_OUTLINE,
                         primary: str = PRIMARY_COLOUR, outline_colour: str = OUTLINE_COLOUR,
                         alignment: int = 2, bold: bool = True) -> str:
    """构造 ASS 的 Style 行。PlayRes=视频分辨率时 FontSize/Margin 均为像素；alignment=2 底部居中。"""
    return (
        f"Style: Default,{font},{font_size},{primary},&H000000FF,{outline_colour},"
        f"&H80000000,{1 if bold else 0},0,0,0,100,100,0,0,1,{outline},0,{alignment},"
        f"{margin_x},{margin_x},{margin_v},1"
    )


def srt_to_ass(srt_path: Path, ass_path: Path, style_line: str,
               playresx: int, playresy: int) -> Path:
    """把 SRT 转成自定 PlayRes 的 ASS：字号按像素、长句按画面宽度折行。"""
    cues = parse_srt(srt_path)
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {playresx}\n"
        f"PlayResY: {playresy}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour,"
        " BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,"
        " BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"{style_line}\n"
        "\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    for start, end, body in cues:
        text = body.replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},"
                     f"Default,,0,0,0,,{text}")
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return ass_path


def _parse_playres(resolution: str) -> tuple:
    m = re.fullmatch(r"\s*(\d{2,5})\s*[xX*×:]\s*(\d{2,5})\s*", resolution or "")
    return (int(m.group(1)), int(m.group(2))) if m else (1080, 1920)


# --------------------------------------------------------------------------- #
# 音频混音滤镜
# --------------------------------------------------------------------------- #
def _narration_chain(label_out: str, volume: float) -> str:
    return (f"[1:a]aresample={AUDIO_SAMPLE_RATE},"
            f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"volume={volume}{label_out}")


def _bgm_chain(bgm_index: int, volume: float, fade_start: Optional[float],
               fade_dur: float, label_out: str) -> str:
    parts = [f"[{bgm_index}:a]aresample={AUDIO_SAMPLE_RATE}",
             "aformat=sample_fmts=fltp:channel_layouts=stereo",
             f"volume={volume}"]
    if fade_start is not None and fade_dur > 0:
        parts.append(f"afade=t=out:st={max(fade_start,0):.3f}:d={fade_dur:.3f}")
    return ",".join(parts) + label_out


def build_audio_filter(has_bgm: bool, bgm_index: int, narration_volume: float,
                       bgm_volume: float, duck: bool, total_duration: float,
                       bgm_fade: float) -> str:
    """构造音频混音滤镜链，输出标签 [a]。"""
    if not has_bgm:
        return _narration_chain("[a]", narration_volume)

    fade_start = max(total_duration - bgm_fade, 0.0) if bgm_fade > 0 else None
    bgm = _bgm_chain(bgm_index, bgm_volume, fade_start, bgm_fade, "[bgm]")
    if duck:
        # 旁白一路直接输出，一路作为侧链控制信号压缩 BGM（人声出现时自动压低音乐）
        narr = _narration_chain(",asplit=2[narr][sc]", narration_volume)
        comp = ("[bgm][sc]sidechaincompress=threshold=0.03:ratio=10:"
                "attack=20:release=300[bgmd]")
        mix = ("[narr][bgmd]amix=inputs=2:duration=first:"
               "dropout_transition=0:normalize=0[mix]")
    else:
        narr = _narration_chain("[narr]", narration_volume)
        mix = ("[narr][bgm]amix=inputs=2:duration=first:"
               "dropout_transition=0:normalize=0[mix]")
    limit = "[mix]alimiter=limit=0.95[a]"
    return ";".join([narr, bgm, comp if duck else "", mix, limit]).replace(";;", ";")


# --------------------------------------------------------------------------- #
# 命令构造
# --------------------------------------------------------------------------- #
def build_command(ffmpeg_bin: str, video_list: Path, audio_list: Path,
                  out_path: Path, *, ass_path: Optional[Path], bgm_path: Optional[Path],
                  narration_volume: float, bgm_volume: float,
                  duck: bool, total_duration: float, bgm_fade: float,
                  crf: int, x264_preset: str, audio_bitrate: str) -> tuple[list[str], str]:
    """构造完整 ffmpeg 命令，返回 (argv, filter_complex)。"""
    argv = [ffmpeg_bin, "-y"]
    # 输入 0：视频 concat；输入 1：配音 concat
    argv += ["-f", "concat", "-safe", "0", "-i", str(video_list)]
    argv += ["-f", "concat", "-safe", "0", "-i", str(audio_list)]
    # 输入 2（可选）：BGM，循环铺满
    bgm_index = None
    if bgm_path:
        bgm_index = 2
        argv += ["-stream_loop", "-1", "-i", str(bgm_path)]

    # 视频链：烧 ASS 字幕（可选）→ 统一像素格式
    vchain = "[0:v]"
    if ass_path:
        ass = str(Path(ass_path).resolve())
        if "'" in ass:
            raise ComposeError(f"ASS 路径含单引号，暂不支持: {ass}")
        vchain += f"ass='{ass}',"
    vchain += "format=yuv420p[v]"

    achain = build_audio_filter(
        has_bgm=bgm_path is not None, bgm_index=bgm_index or 2,
        narration_volume=narration_volume, bgm_volume=bgm_volume, duck=duck,
        total_duration=total_duration, bgm_fade=bgm_fade,
    )
    filter_complex = f"{vchain};{achain}"

    argv += [
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", x264_preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", audio_bitrate, "-ar", str(AUDIO_SAMPLE_RATE),
        "-shortest", "-movflags", "+faststart",
        str(out_path),
    ]
    return argv, filter_complex


def probe_media(ffmpeg_bin: str, path: Path) -> tuple[Optional[float], str]:
    """读取成片的时长与流信息摘要。"""
    duration = None
    streams = ""
    try:
        out = subprocess.run([ffmpeg_bin, "-hide_banner", "-i", str(path)],
                             capture_output=True, text=True, timeout=30)
        err = out.stderr or ""
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", err)
        if m:
            duration = round(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)), 3)
        v = re.search(r"Stream.*Video:\s*([^,]+).*,\s*(\d+x\d+)", err)
        a = re.search(r"Stream.*Audio:\s*([^,]+),\s*(\d+)\s*Hz,\s*(\w+)", err)
        parts = []
        if v:
            parts.append(f"video={v.group(1).strip()} {v.group(2)}")
        if a:
            parts.append(f"audio={a.group(1).strip()} {a.group(2)}Hz {a.group(3)}")
        streams = " | ".join(parts)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return duration, streams


# --------------------------------------------------------------------------- #
# 合成主流程
# --------------------------------------------------------------------------- #
def compose(video_paths: list, audio_paths: list, ffmpeg_bin: str, *, output_dir: Path,
            total_duration: float, srt_path: Optional[Path], style_line: Optional[str],
            bgm_path: Optional[Path], narration_volume: float, bgm_volume: float,
            duck: bool, bgm_fade: float, crf: int, x264_preset: str,
            audio_bitrate: str, resolution: str, fps: int,
            dry_run: bool = False, keep_intermediate: bool = False) -> ComposeResult:
    """拼接视频 + 混音 + 烧字幕，输出 final.mp4。

    video_paths / audio_paths 分别为视频段与配音段的有序列表；两者段数可不同
    （Shot 模式：视频=镜头片段，配音=场景音频），只要总时长一致即可对齐。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final.mp4"
    n_video = len(video_paths)

    video_list = write_concat_list(list(video_paths),
                                   output_dir / "video_concat.txt")
    audio_list = write_concat_list(list(audio_paths),
                                   output_dir / "audio_concat.txt")

    # 字幕：SRT → ASS（自定 PlayRes，像素级字号 + 按画面宽度折行）
    ass_path: Optional[Path] = None
    if srt_path:
        px, py = _parse_playres(resolution)
        ass_path = srt_to_ass(srt_path, output_dir / "subtitle.ass",
                              style_line or build_ass_style_line(), px, py)

    argv, filter_complex = build_command(
        ffmpeg_bin, video_list, audio_list, final_path,
        ass_path=ass_path, bgm_path=bgm_path,
        narration_volume=narration_volume, bgm_volume=bgm_volume, duck=duck,
        total_duration=total_duration, bgm_fade=bgm_fade,
        crf=crf, x264_preset=x264_preset, audio_bitrate=audio_bitrate,
    )

    if dry_run:
        logger.info("[dry-run] filter_complex:\n%s", filter_complex)
        logger.debug("[dry-run] cmd: %s", " ".join(_shell_quote(a) for a in argv))
        return ComposeResult(
            output_path=final_path, duration_seconds=None, file_size=0,
            scene_count=n_video, resolution=resolution, has_bgm=bgm_path is not None,
            has_subtitle=srt_path is not None, elapsed_seconds=0.0,
            ffmpeg_bin=ffmpeg_bin, command=argv,
        )

    logger.info("→ 合成中：%d 段视频 / %d 段配音，BGM=%s，字幕=%s",
                n_video, len(audio_paths), "有" if bgm_path else "无", "有" if srt_path else "无")
    logger.debug("filter_complex: %s", filter_complex)
    started = time.perf_counter()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        raise ComposeError(f"合成超时: {exc}") from exc
    except OSError as exc:
        raise ComposeError(f"调用 ffmpeg 失败: {exc}") from exc
    elapsed = round(time.perf_counter() - started, 2)

    if proc.returncode != 0 or not final_path.exists():
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise ComposeError(f"ffmpeg 合成失败 (rc={proc.returncode}):\n  " + "\n  ".join(tail))

    if not keep_intermediate:
        for tmp in (video_list, audio_list):
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    duration, streams = probe_media(ffmpeg_bin, final_path)
    size = final_path.stat().st_size
    logger.info("✓ 合成完成: %s", final_path)
    logger.info("  时长: %s | 体积: %.2f MB | 耗时: %.1fs",
                f"{duration:.3f}s" if duration else "未知", size / 1024 / 1024, elapsed)
    if streams:
        logger.info("  流: %s", streams)

    return ComposeResult(
        output_path=final_path, duration_seconds=duration, file_size=size,
        scene_count=n_video, resolution=resolution, has_bgm=bgm_path is not None,
        has_subtitle=srt_path is not None, elapsed_seconds=elapsed,
        ffmpeg_bin=ffmpeg_bin, command=argv,
    )


def _shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'" if re.search(r"[^\w@%+=:,./-]", text) else text


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def load_project_timeline(project_dir: str | Path) -> tuple:
    """短剧项目时间轴：按 script.json 镜头顺序返回 1:1 的 (video_paths, audio_paths, total)。

    视频=video/<shot>.mp4（装配段），音频=audio/<shot>.mp3（含纯视觉镜头的静音）。
    """
    pdir = Path(project_dir).expanduser().resolve()
    script = project_io.load_script(pdir)
    video_paths: list = []
    audio_paths: list = []
    total = 0.0
    for scene, shot in script.iter_shots():
        vp = project_io.segment_path(pdir, shot.id)
        ap = project_io.audio_path(pdir, shot.id)
        if not vp.exists():
            raise ComposeError(f"镜头 {shot.id} 找不到视频段 {vp}（请先跑 03-video）")
        if not ap.exists():
            raise ComposeError(f"镜头 {shot.id} 找不到音频 {ap}（请先跑 01-voice）")
        video_paths.append(vp)
        audio_paths.append(ap)
        total += float(shot.runtime_sec if shot.runtime_sec is not None else shot.duration_sec)
    if not video_paths:
        raise ComposeError("script.json 中没有镜头")
    return video_paths, audio_paths, round(total, 3)


def _compose_project(args: argparse.Namespace, ffmpeg_bin: str) -> int:
    """短剧项目合成：逐 shot 音视频 1:1 拼接 + 混音 + 烧字幕 → projects/<id>/final.mp4。"""
    pdir = Path(args.project).expanduser().resolve()
    if not (pdir / "script.json").exists():
        raise ComposeError(f"项目缺少 script.json: {pdir}（请先跑 00-scripts / 01-voice）")
    script = project_io.load_script(pdir)
    video_paths, audio_paths, total = load_project_timeline(pdir)

    srt_path: Optional[Path] = None
    if not args.no_subtitle:
        if args.srt:
            srt_path = Path(args.srt).expanduser().resolve()
            if not srt_path.exists():
                raise ComposeError(f"字幕文件不存在: {srt_path}")
        else:
            cand = project_io.subtitle_path(pdir)
            if cand.exists():
                srt_path = cand.resolve()
            else:
                logger.warning("未找到 subtitle.srt（%s），本次不烧字幕", cand)
    style_line = build_ass_style_line(
        font=args.font, font_size=args.font_size, margin_v=args.margin_v,
        margin_x=args.margin_x) if srt_path is not None else None

    bgm_path: Optional[Path] = None
    if args.bgm:
        bgm_path = Path(args.bgm).expanduser().resolve()
        if not bgm_path.exists():
            raise ComposeError(f"BGM 文件不存在: {bgm_path}")

    resolution = prompt_kit.resolution_for(script)
    out_dir = Path(args.output).expanduser() if args.output else pdir
    logger.info("ffmpeg    : %s", ffmpeg_bin)
    logger.info("模式      : 短剧项目（%d 镜头，音视频 1:1）", len(video_paths))
    logger.info("预计总时长: %.3fs", total)
    logger.info("字幕      : %s", srt_path or "无")
    logger.info("BGM       : %s%s", bgm_path or "无", "（侧链闪避）" if bgm_path and args.duck else "")

    result = compose(
        video_paths, audio_paths, ffmpeg_bin, output_dir=out_dir, total_duration=total,
        srt_path=srt_path, style_line=style_line, bgm_path=bgm_path,
        narration_volume=args.narration_volume, bgm_volume=args.bgm_volume,
        duck=args.duck, bgm_fade=args.bgm_fade, crf=args.crf,
        x264_preset=args.x264_preset, audio_bitrate=args.audio_bitrate,
        resolution=resolution, fps=30, dry_run=args.dry_run,
        keep_intermediate=args.keep_intermediate,
    )
    if args.dry_run:
        logger.info("[dry-run] 目标输出: %s", result.output_path)
    else:
        logger.info("— 结果 —")
        logger.info("  成片    : %s", result.output_path.resolve())
        logger.info("  时长    : %s", f"{result.duration_seconds:.3f}s" if result.duration_seconds else "未知")
        logger.info("  体积    : %.2f MB", result.file_size / 1024 / 1024)
        logger.info("  分辨率  : %s", result.resolution)
    print(json.dumps({
        "ok": True, "mode": "project", "dry_run": args.dry_run,
        "output_path": str(result.output_path.resolve()),
        "duration_seconds": result.duration_seconds, "file_size": result.file_size,
        "shot_count": len(video_paths), "has_bgm": result.has_bgm,
        "has_subtitle": result.has_subtitle, "resolution": result.resolution,
    }, ensure_ascii=False))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compose.generate",
        description="拼接视频 + 混音（配音+BGM）+ 烧录字幕 → final.mp4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project", required=True,
                        help="短剧项目目录（projects/<id>）：读 script.json，逐 shot 音视频 1:1 拼接，成片写项目目录")
    parser.add_argument("--srt", help="字幕 SRT 路径（默认取项目目录 subtitle.srt）")
    parser.add_argument("--no-subtitle", action="store_true", help="不烧录字幕")
    parser.add_argument("--bgm", help="背景音乐路径（可选；自动循环铺满并压低音量）")
    parser.add_argument("--narration-volume", type=float, default=DEFAULT_NARRATION_VOLUME, help="旁白音量")
    parser.add_argument("--bgm-volume", type=float, default=DEFAULT_BGM_VOLUME, help="BGM 音量")
    parser.add_argument("--duck", action="store_true", help="对 BGM 做侧链闪避（人声出现时自动压低音乐）")
    parser.add_argument("--bgm-fade", type=float, default=DEFAULT_BGM_FADE, help="BGM 结尾淡出秒数（0 关闭）")
    parser.add_argument("--font", default=DEFAULT_FONT, help="字幕字体名（需系统已安装）")
    parser.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE, help="字幕字号（像素，基于视频分辨率）")
    parser.add_argument("--margin-v", type=int, default=DEFAULT_MARGIN_V, help="字幕底部边距（像素）")
    parser.add_argument("--margin-x", type=int, default=DEFAULT_MARGIN_X, help="字幕左右安全边距（像素）")
    parser.add_argument("--crf", type=int, default=20, help="libx264 CRF 质量")
    parser.add_argument("--preset", dest="x264_preset", default="medium", help="libx264 编码预设")
    parser.add_argument("--audio-bitrate", default="192k", help="AAC 音频码率")
    parser.add_argument("--output", help="输出目录（默认写项目目录）")
    parser.add_argument("--ffmpeg", help="显式指定 ffmpeg 可执行文件路径")
    parser.add_argument("--keep-intermediate", action="store_true", help="保留 concat 列表等中间文件")
    parser.add_argument("--dry-run", action="store_true", help="只构造并打印命令，不实际合成")
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
        ffmpeg_bin = resolve_ffmpeg(args.ffmpeg)
        return _compose_project(args, ffmpeg_bin)

    except FFmpegNotFound as exc:
        logger.error("%s", exc)
        print(json.dumps({"ok": False, "error_type": "FFmpegNotFound", "error": str(exc)}, ensure_ascii=False))
        return 5
    except ComposeError as exc:
        logger.error("合成失败: %s", exc)
        print(json.dumps({"ok": False, "error_type": "ComposeError", "error": str(exc)}, ensure_ascii=False))
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
