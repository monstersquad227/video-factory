#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""字幕生成模块（流水线 step 4）。

读取短剧项目 ``projects/<id>/script.json``，逐 shot 在其真实时间窗 ``[start, end]`` 内，
把 ``narration``（旁白）与 ``dialogue``（对白，可加「角色名：」前缀）断句、按字数比例
分配时间，输出标准 SRT 字幕到 ``projects/<id>/subtitle.srt``。

设计要点：
- 时间轴来自 01-voice 回写的 shot.start/end（真实配音时长），字幕天然与音画同步；
- 断句：先按标点切子句，再按「单行最大字数」贪心合并，超长子句硬切；
- 时间分配：同一 shot 窗口按每条字幕字数比例切分（TTS 语速近似恒定）；
- 纯视觉镜头（无 narration/dialogue）不产字幕。

CLI 示例::

    python 04-subtitle/generate.py --project projects/the-last-track
    python 04-subtitle/generate.py --project projects/the-last-track --no-speaker --max-chars 14
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import project as project_io  # noqa: E402


logger = logging.getLogger("subtitle.generate")

# 断句标点分级：强断句（句末，倾向于在此换行）与弱断句（句内，可合并）
_STRONG_PUNCT = "。！？!?；;…\n"
_WEAK_PUNCT = "，、：,:"
_ALL_PUNCT = _STRONG_PUNCT + _WEAK_PUNCT

DEFAULT_MAX_CHARS = 16          # 竖屏 9:16 单行建议 12~18 个汉字
DEFAULT_MIN_DURATION = 0.4      # 单条字幕最小可读时长（秒）


# --------------------------------------------------------------------------- #
# 异常与数据结构
# --------------------------------------------------------------------------- #
class SubtitleError(Exception):
    """字幕生成基础异常。"""


@dataclass
class Cue:
    """一条字幕。"""

    index: int
    start: float
    end: float
    text: str


# --------------------------------------------------------------------------- #
# 断句
# --------------------------------------------------------------------------- #
def _split_clauses(text: str) -> list[str]:
    """按所有标点把文本切成子句（保留标点，去空白）。"""
    clauses: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in _ALL_PUNCT:
            clauses.append(buf)
            buf = ""
    if buf.strip():
        clauses.append(buf)
    return [c.strip() for c in clauses if c.strip()]


def _hard_split(text: str, max_chars: int) -> list[str]:
    """无标点的超长片段按固定字数硬切。"""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def split_narration(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """把一段文本断成若干条字幕文本。

    规则：
    1. 先按标点切子句；
    2. 贪心合并相邻子句，直到超过 ``max_chars``；
    3. 遇到句末强标点且当前串已达半行以上，主动换行（读起来更自然）；
    4. 单个子句本身就超长时按字数硬切。
    """
    if max_chars < 4:
        raise SubtitleError(f"max_chars 过小: {max_chars}")
    # 容差：单条子句只超出 max_chars 一点点时整条保留，避免把词/数字切断；
    # 仅当明显超长（超过阈值）才按字数硬切。
    hard_split_at = max_chars + max(4, max_chars // 4)
    clauses = _split_clauses(text)
    cues: list[str] = []
    cur = ""
    for clause in clauses:
        if len(clause) > hard_split_at:        # 明显超长子句：先落盘当前，再硬切
            if cur:
                cues.append(cur)
                cur = ""
            cues.extend(_hard_split(clause, max_chars))
            continue
        if cur and len(cur) + len(clause) > max_chars:
            cues.append(cur)
            cur = clause
        else:
            cur += clause
        # 句末强标点且已积累到半行以上 → 断行
        if cur and cur[-1] in _STRONG_PUNCT and len(cur) >= max_chars * 0.5:
            cues.append(cur)
            cur = ""
    if cur:
        cues.append(cur)
    return cues


# --------------------------------------------------------------------------- #
# 时间分配
# --------------------------------------------------------------------------- #
def _content_len(text: str) -> int:
    """用于加权的“内容字数”：忽略标点与空白，只数实际发声字符。"""
    return sum(1 for ch in text if ch not in _ALL_PUNCT and not ch.isspace())


def distribute_time(segments: list[str], start: float, end: float,
                    min_duration: float = DEFAULT_MIN_DURATION) -> list[tuple[str, float, float]]:
    """把 ``[start, end]`` 时间窗按各段字数比例切分，返回 (文本, 起, 止) 列表。"""
    total = max(end - start, 0.0)
    weights = [max(_content_len(s), 1) for s in segments]
    wsum = sum(weights)
    out: list[tuple[str, float, float]] = []
    cursor = start
    last = len(segments) - 1
    for i, (seg, w) in enumerate(zip(segments, weights)):
        seg_start = cursor
        if i == last:
            seg_end = end
        else:
            seg_end = cursor + total * w / wsum
            # 保证最小可读时长，但不越过窗口终点
            seg_end = min(max(seg_end, seg_start + min_duration), end)
        out.append((seg, round(seg_start, 3), round(seg_end, 3)))
        cursor = seg_end
    return out


# --------------------------------------------------------------------------- #
# SRT 组装
# --------------------------------------------------------------------------- #
def format_timestamp(seconds: float) -> str:
    """秒 → SRT 时间戳 ``HH:MM:SS,mmm``。"""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def render_srt(cues: list[Cue]) -> str:
    """把 cue 列表渲染成 SRT 文本。"""
    blocks = [
        f"{c.index}\n{format_timestamp(c.start)} --> {format_timestamp(c.end)}\n{c.text}"
        for c in cues
    ]
    return "\n\n".join(blocks) + "\n"


def write_srt(cues: list[Cue], out_path: str | Path) -> Path:
    """写出 SRT 文件（UTF-8）。"""
    p = Path(out_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(render_srt(cues), encoding="utf-8")
    except OSError as exc:
        raise SubtitleError(f"写入 SRT 失败: {exc}") from exc
    return p


# --------------------------------------------------------------------------- #
# 短剧项目：script.json 的 narration + dialogue → cues
# --------------------------------------------------------------------------- #
def load_project_subtitle_items(project_dir: str | Path, speaker_prefix: bool = True) -> list:
    """从 script.json 读取每个 shot 的 (start, end, [文本段])：narration + dialogue。"""
    pdir = Path(project_dir).expanduser().resolve()
    script = project_io.load_script(pdir)
    items: list = []
    for scene, shot in script.iter_shots():
        if shot.start is None or shot.end is None:
            logger.warning("shot %s 缺少时间轴，跳过（请先跑 01-voice --project）", shot.id)
            continue
        segs: list = []
        if shot.narration.strip():
            segs.append(shot.narration.strip())
        for line in shot.dialogue:
            txt = (line.text or "").strip()
            if not txt:
                continue
            if speaker_prefix:
                c = script.characters.get(line.speaker)
                name = c.name if c else line.speaker
                segs.append(f"{name}：{txt}")
            else:
                segs.append(txt)
        if segs:
            items.append((float(shot.start), float(shot.end), segs))
    if not items:
        raise SubtitleError("script.json 中没有带时间轴与文本的镜头")
    return items


def build_cues_from_items(items: list, max_chars: int = DEFAULT_MAX_CHARS,
                          min_duration: float = DEFAULT_MIN_DURATION) -> list[Cue]:
    """把 [(start,end,[文本段])] 转成全局连续编号的 cue 列表。"""
    cues: list[Cue] = []
    idx = 1
    for start, end, segs in items:
        texts: list[str] = []
        for seg in segs:
            texts.extend(split_narration(seg, max_chars=max_chars))
        if not texts:
            continue
        for text, s, e in distribute_time(texts, start, end, min_duration=min_duration):
            cues.append(Cue(index=idx, start=s, end=e, text=text))
            idx += 1
    return cues


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtitle.generate",
        description="字幕生成：读 projects/<id>/script.json 的 narration+dialogue → subtitle.srt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project", required=True,
                        help="短剧项目目录（projects/<id>）：读 script.json，写 subtitle.srt")
    parser.add_argument("--no-speaker", action="store_true", help="对白字幕不加「角色名：」前缀")
    parser.add_argument("--output", help="SRT 输出路径（默认写到项目目录 subtitle.srt）")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                        help="单条字幕最大字数")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION,
                        help="单条字幕最小可读时长（秒）")
    parser.add_argument("--stdout", action="store_true", help="同时把 SRT 打印到标准输出")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        pdir = Path(args.project).expanduser().resolve()
        items = load_project_subtitle_items(pdir, speaker_prefix=not args.no_speaker)
        cues = build_cues_from_items(items, max_chars=args.max_chars,
                                     min_duration=args.min_duration)
        if not cues:
            raise SubtitleError("未生成任何字幕（旁白/对白为空？）")

        out_path = Path(args.output).expanduser() if args.output \
            else project_io.subtitle_path(pdir)
        written = write_srt(cues, out_path)

        total = cues[-1].end if cues else 0.0
        logger.info("— 结果 —")
        logger.info("  有字幕镜头: %d", len(items))
        logger.info("  字幕条数  : %d", len(cues))
        logger.info("  时间轴终点: %.3fs", total)
        logger.info("  SRT 文件  : %s", written.resolve())
        if args.stdout:
            print(render_srt(cues))
        print(json.dumps({
            "ok": True,
            "srt_path": str(written.resolve()),
            "cue_count": len(cues),
            "source_count": len(items),
            "timeline_end": round(total, 3),
        }, ensure_ascii=False))
        return 0
    except SubtitleError as exc:
        logger.error("字幕生成失败: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("用户中断")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("未预期错误: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
