#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键编排：一段话 → projects/<id>/final.mp4。

按顺序调用六个模块，全部读写同一个项目目录 ``projects/<id>/``（script.json 为唯一数据源）：

    00-scripts  一段话 → script.json（project/characters/scenes[].shots[]，模板 schema）
    01-voice    逐 shot 多角色配音（旁白+对白）→ audio/ + 回写真实时长/时间轴
    02-image    角色定妆照 + 逐 shot 关键帧（多图参考）→ assets/chars/ + keyframes/
    03-video    wan3.0-video 图生视频 + 逐 shot 定长装配 → video/ai_*.mp4 + video/*.mp4
    04-subtitle narration + dialogue → subtitle.srt
    05-compose  逐 shot 音视频 1:1 拼接 + 混音 + 烧字幕 → final.mp4

任一步失败即停止。各模块自带断点续跑（已存在的音频/图片/AI片段会跳过），
中断后用同一个 ``--project`` 重跑即可从断点继续。

CLI 示例::

    # 全量（含 AI 视频）
    python run.py --text "一个关于坚持与告别的短剧" --project-id the-last-track --target-duration 70

    # 省钱快跑：跳过 AI 视频（所有镜头用关键帧 Ken Burns 运镜）
    python run.py --text "..." --target-duration 30 --skip-ai

    # 断点续跑已有项目
    python run.py --project projects/the-last-track
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
PY = sys.executable
PROJECTS_ROOT = ROOT / "projects"

DEFAULT_TARGET_DURATION = 70.0
DEFAULT_ASPECT = "9:16"
STEPS = ["scripts", "voice", "image", "video", "subtitle", "compose"]


def slugify(text: str, fallback: str = "episode") -> str:
    slug = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", (text or "").strip()).strip("-")
    return (slug[:40] or fallback)


def run_step(title: str, argv: list) -> None:
    """执行一个模块步骤，失败即中止整个流水线。"""
    print(f"\n{'='*66}\n▶ {title}\n  $ {' '.join(str(a) for a in argv)}\n{'='*66}", flush=True)
    proc = subprocess.run([str(a) for a in argv], cwd=str(ROOT))
    if proc.returncode != 0:
        print(f"\n[ERROR] 步骤失败：{title}（exit {proc.returncode}）", file=sys.stderr)
        print("        修复后可用同一个 --project 重跑以从断点继续。", file=sys.stderr)
        sys.exit(proc.returncode)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="一键编排：一段话 → projects/<id>/final.mp4（00-scripts → 05-compose）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", help="主题/故事（一段话）")
    src.add_argument("--text-file", help="从文件读取主题/故事")
    p.add_argument("--project", help="项目目录（默认 projects/<project-id>；传已有目录可断点续跑）")
    p.add_argument("--project-id", help="项目 id（默认由文案 slug 化）")
    p.add_argument("--target-duration", type=float, default=DEFAULT_TARGET_DURATION, help="目标总时长（秒）")
    p.add_argument("--aspect", default=DEFAULT_ASPECT, help="画幅，如 9:16")
    p.add_argument("--style", default="", help="统一画面风格提示")
    p.add_argument("--language", default="zh-CN", help="语言")
    p.add_argument("--script-model", help="脚本 LLM 模型（覆盖 00-scripts/.env）")
    p.add_argument("--narrator-voice", help="旁白音色 ID（传给 01-voice）")
    p.add_argument("--image-seed", type=int, help="关键帧固定 seed（提升人物一致性）")
    p.add_argument("--ai-concurrency", type=int, help="AI 视频并发数")
    p.add_argument("--ai-resolution", choices=["480P", "720P", "1080P"], help="AI 视频分辨率档位")
    p.add_argument("--skip-ai", action="store_true",
                   help="跳过 AI 视频生成（03-video 只装配，所有镜头用关键帧 Ken Burns）")
    p.add_argument("--bgm", help="背景音乐路径（传给 05-compose，自动侧链闪避）")
    p.add_argument("--force-scripts", action="store_true", help="即使 script.json 已存在也重新生成")
    p.add_argument("--stop-after", choices=STEPS, default="compose", help="运行到某一步为止（分阶段调试）")
    args = p.parse_args(argv)

    text = ""
    if args.text:
        text = args.text
    elif args.text_file:
        tf = Path(args.text_file).expanduser()
        if not tf.exists():
            print(f"[ERROR] 文案文件不存在: {tf}", file=sys.stderr)
            return 2
        text = tf.read_text(encoding="utf-8").strip()

    project_id = args.project_id or (slugify(text) if text else "")
    if args.project:
        project_dir = Path(args.project).expanduser().resolve()
    else:
        if not project_id:
            print("[ERROR] 需要 --text / --text-file，或 --project / --project-id", file=sys.stderr)
            return 2
        project_dir = PROJECTS_ROOT / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    if not project_id:
        project_id = project_dir.name
    print(f"项目目录: {project_dir}")

    stop_at = STEPS.index(args.stop_after)
    script_json = project_dir / "script.json"

    # 0) 脚本生成
    if stop_at >= 0:
        if script_json.exists() and not args.force_scripts:
            print("\n▶ 00-scripts：复用已存在的 script.json（--force-scripts 可重新生成）")
        else:
            if not text.strip():
                print("[ERROR] 缺少 script.json 且未提供 --text，无法生成脚本", file=sys.stderr)
                return 2
            cmd = [PY, "00-scripts/generate.py", "--text", text, "--project", str(project_dir),
                   "--project-id", project_id, "--target-duration", str(args.target_duration),
                   "--aspect", args.aspect, "--language", args.language]
            if args.style:
                cmd += ["--style", args.style]
            if args.script_model:
                cmd += ["--model", args.script_model]
            run_step("00-scripts 脚本生成（project + characters + scenes/shots）", cmd)

    # 1) 配音 + 时间轴
    if stop_at >= 1:
        cmd = [PY, "01-voice/generate.py", "--project", str(project_dir)]
        if args.narrator_voice:
            cmd += ["--narrator-voice", args.narrator_voice]
        run_step("01-voice 逐 shot 多角色配音 + 时间轴回写", cmd)

    # 2) 定妆照 + 关键帧
    if stop_at >= 2:
        cmd = [PY, "02-image/generate.py", "--project", str(project_dir)]
        if args.image_seed is not None:
            cmd += ["--consistency-seed", str(args.image_seed)]
        run_step("02-image 角色定妆照 + 逐 shot 关键帧", cmd)

    # 3) AI 视频 + 装配
    if stop_at >= 3:
        stage = "assemble" if args.skip_ai else "all"
        cmd = [PY, "03-video/generate.py", "--project", str(project_dir), "--stage", stage]
        if args.ai_concurrency is not None:
            cmd += ["--ai-concurrency", str(args.ai_concurrency)]
        if args.ai_resolution:
            cmd += ["--ai-resolution", args.ai_resolution]
        run_step(f"03-video 视频生成与装配（stage={stage}）", cmd)

    # 4) 字幕
    if stop_at >= 4:
        run_step("04-subtitle 字幕（旁白 + 对白）",
                 [PY, "04-subtitle/generate.py", "--project", str(project_dir)])

    # 5) 合成
    if stop_at >= 5:
        cmd = [PY, "05-compose/generate.py", "--project", str(project_dir)]
        if args.bgm:
            cmd += ["--bgm", args.bgm, "--duck"]
        run_step("05-compose 合成成片", cmd)

    final = project_dir / "final.mp4"
    print(f"\n{'='*66}")
    if stop_at >= 5 and final.exists():
        print(f"✅ 完成：{final}")
    else:
        print(f"✅ 已运行到 '{args.stop_after}' 步骤；项目目录：{project_dir}")
    print(f"{'='*66}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
