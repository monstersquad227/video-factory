#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""项目（run 目录）IO 与状态回写：projects/<id>/script.json 为唯一数据源。

约定目录布局::

    projects/<id>/
      script.json            唯一数据源（含 status/outputs 回写）
      assets/chars/<key>.png 角色定妆照
      keyframes/<shot>.png   每 shot 关键帧（首帧）
      video/ai_<shot>.mp4    wan3.0-video 图生视频原片
      video/<shot>.mp4       定长装配段
      audio/<shot>.mp3       每 shot 音频（narration+dialogue）
      subtitle.srt
      final.mp4

outputs.* 一律存**相对项目目录**的路径，便于整体移动/复现。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Union

from .schema import Script

PathLike = Union[str, Path]


def _safe(name: str) -> str:
    """把 shot_id / 角色 key 变成安全文件名片段。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-._") or "x"


# --------------------------------------------------------------------------- #
# 定位 script.json / 项目目录
# --------------------------------------------------------------------------- #
def resolve_script_path(arg: PathLike) -> Path:
    """接受项目目录或 script.json 路径，返回 script.json 的绝对路径。"""
    p = Path(arg).expanduser().resolve()
    if p.is_dir():
        cand = p / "script.json"
        if not cand.exists():
            raise FileNotFoundError(f"项目目录缺少 script.json: {cand}")
        return cand
    if not p.exists():
        raise FileNotFoundError(f"script.json 不存在: {p}")
    return p


def project_dir(script_path: PathLike) -> Path:
    return resolve_script_path(script_path).parent


# --------------------------------------------------------------------------- #
# 目录 / 文件路径助手（返回绝对路径）
# --------------------------------------------------------------------------- #
def chars_dir(pdir: PathLike) -> Path:
    return Path(pdir) / "assets" / "chars"


def keyframes_dir(pdir: PathLike) -> Path:
    return Path(pdir) / "keyframes"


def video_dir(pdir: PathLike) -> Path:
    return Path(pdir) / "video"


def audio_dir(pdir: PathLike) -> Path:
    return Path(pdir) / "audio"


def subtitle_path(pdir: PathLike) -> Path:
    return Path(pdir) / "subtitle.srt"


def final_path(pdir: PathLike) -> Path:
    return Path(pdir) / "final.mp4"


def char_portrait_path(pdir: PathLike, key: str) -> Path:
    return chars_dir(pdir) / f"{_safe(key)}.png"


def keyframe_path(pdir: PathLike, shot_id: str) -> Path:
    return keyframes_dir(pdir) / f"{_safe(shot_id)}.png"


def ai_clip_path(pdir: PathLike, shot_id: str) -> Path:
    return video_dir(pdir) / f"ai_{_safe(shot_id)}.mp4"


def segment_path(pdir: PathLike, shot_id: str) -> Path:
    return video_dir(pdir) / f"{_safe(shot_id)}.mp4"


def audio_path(pdir: PathLike, shot_id: str) -> Path:
    return audio_dir(pdir) / f"{_safe(shot_id)}.mp3"


def ensure_dirs(pdir: PathLike) -> None:
    for d in (chars_dir(pdir), keyframes_dir(pdir), video_dir(pdir), audio_dir(pdir)):
        d.mkdir(parents=True, exist_ok=True)


def rel(pdir: PathLike, path: PathLike) -> str:
    """把绝对路径转为相对项目目录的 posix 字符串（写入 outputs.*）。"""
    try:
        return Path(path).resolve().relative_to(Path(pdir).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


# --------------------------------------------------------------------------- #
# 读写 script.json
# --------------------------------------------------------------------------- #
def load_script(path: PathLike) -> Script:
    """读取并用 Pydantic 校验 script.json。"""
    sp = resolve_script_path(path)
    try:
        raw = sp.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"读取 script.json 失败: {exc}") from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise RuntimeError(f"script.json 不是合法 JSON: {exc}") from exc
    return Script.model_validate(data)


def save_script(script: Script, path: PathLike) -> Path:
    """原子写回 script.json（Pydantic 序列化，保留运行期字段）。"""
    sp = resolve_script_path(path) if Path(path).exists() else Path(path).expanduser().resolve()
    sp.parent.mkdir(parents=True, exist_ok=True)
    data = script.model_dump(mode="json", exclude_none=False)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    fd, tmp = tempfile.mkstemp(dir=str(sp.parent), prefix=".script.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, sp)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return sp


def load_raw(path: PathLike) -> dict:
    """读取原始 dict（不做 Pydantic 校验，供 00-scripts 修复流程用）。"""
    sp = resolve_script_path(path)
    return json.loads(sp.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 状态回写
# --------------------------------------------------------------------------- #
def set_shot_status(script: Script, shot_id: str, *,
                    keyframe: Optional[str] = None, video: Optional[str] = None,
                    audio: Optional[str] = None) -> None:
    st = script.get_shot(shot_id)
    if not st:
        return
    if keyframe is not None:
        st.status.keyframe = keyframe  # type: ignore[assignment]
    if video is not None:
        st.status.video = video        # type: ignore[assignment]
    if audio is not None:
        st.status.audio = audio        # type: ignore[assignment]


def set_shot_output(script: Script, shot_id: str, *,
                    keyframe: Optional[str] = None, video: Optional[str] = None,
                    audio: Optional[str] = None) -> None:
    st = script.get_shot(shot_id)
    if not st:
        return
    if keyframe is not None:
        st.outputs.keyframe = keyframe
    if video is not None:
        st.outputs.video = video
    if audio is not None:
        st.outputs.audio = audio
