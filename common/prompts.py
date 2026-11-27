#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提示词模板拼接：script.json 的结构化字段 + prompt_templates → 生图/生视频提示词。

设计原则（来自 script_template.json）：**JSON 是唯一数据源，提示词由代码用模板拼接，
不要手写进镜头里**。这样保证同一角色的 appearance/outfit 原样复用，跨镜头一致。
"""

from __future__ import annotations

from .schema import ASPECT_TO_RESOLUTION, Script, Scene, Shot

# shot_type → (中文, 英文) 景别短语
SHOT_TYPE_PHRASE = {
    "wide": ("全景", "wide establishing shot"),
    "full": ("全景", "full body shot"),
    "medium": ("中景", "medium shot"),
    "close_up": ("特写", "close-up"),
    "extreme_close_up": ("大特写", "extreme close-up detail"),
}

# shot_type / motion 关键词 → Ken Burns 动效预设（回退或静态镜头用）
_MOTION_KEYWORDS = [
    (("拉远", "拉出", "后退", "pull", "zoom out"), "zoom_out"),
    (("推进", "推近", "靠近", "push", "zoom in"), "zoom_in"),
    (("左移", "向左", "pan left"), "pan_left"),
    (("右移", "向右", "横移", "pan right"), "pan_right"),
    (("上摇", "上升", "tilt up"), "pan_up"),
    (("下摇", "下降", "俯", "tilt down"), "pan_down"),
    (("固定", "静止", "static", "lock"), "static"),
]


def shot_type_phrase(shot_type: str) -> str:
    zh, en = SHOT_TYPE_PHRASE.get(shot_type, ("中景", "medium shot"))
    return f"{zh} {en}"


def characters_desc(script: Script, shot: Shot) -> str:
    """把 shot 涉及的角色展开为「名字（外貌；服装）」，appearance/outfit 原样复用。"""
    parts: list[str] = []
    for key in shot.characters:
        c = script.characters.get(key)
        if not c:
            continue
        detail = "；".join(x for x in (c.appearance, c.outfit) if x and x.strip())
        parts.append(f"{c.name}（{detail}）" if detail else c.name)
    return "，".join(parts)


def build_keyframe_prompt(script: Script, scene: Scene, shot: Shot) -> str:
    """按 prompt_templates.keyframe 拼接首帧/关键帧提示词。

    wan2.7-image 无 negative_prompt 参数，故把 project.negative 以「避免：…」并入文本；
    人物一致性主要靠多图参考（定妆照）+ appearance/outfit 原样复用。
    """
    tpl = script.prompt_templates.keyframe
    body = tpl.format(
        style=script.project.style or "",
        shot_type=shot_type_phrase(shot.shot_type),
        location=scene.location or "",
        time_of_day=scene.time_of_day or "",
        characters_desc=characters_desc(script, shot),
        action=shot.action or "",
    )
    body = "，".join(seg for seg in body.split("，") if seg.strip())
    aspect = script.project.aspect_ratio or "9:16"
    tags = [f"vertical {aspect}" if aspect == "9:16" else f"aspect {aspect}",
            "cinematic, photorealistic, high detail",
            "consistent character appearance, outfit and lighting"]
    if script.project.negative:
        tags.append(f"避免：{script.project.negative}")
    tags.append("no readable text, no watermark")
    return f"{body}。" + ", ".join(tags)


def build_video_prompt(script: Script, scene: Scene, shot: Shot,
                       duration_sec: float | None = None) -> str:
    """按 prompt_templates.video 拼接图生视频提示词（首帧已定外观，聚焦动作+运镜）。"""
    tpl = script.prompt_templates.video
    dur = duration_sec if duration_sec is not None else (shot.runtime_sec or shot.duration_sec)
    try:
        dur_val: object = int(round(float(dur)))
    except (TypeError, ValueError):
        dur_val = int(round(shot.duration_sec))
    body = tpl.format(motion=shot.motion or shot.action or "", duration_sec=dur_val)
    aspect = script.project.aspect_ratio or "9:16"
    return (f"{body}。动作自然流畅，人物外观、发型与服装保持一致，电影质感，"
            f"竖屏 {aspect}" if aspect == "9:16" else
            f"{body}。动作自然流畅，人物外观、发型与服装保持一致，电影质感，画幅 {aspect}")


def build_character_prompt(script: Script, key: str) -> str:
    """角色定妆照提示词（文生图，无参考）：固定外貌+服装+统一风格，中性背景、正面半身。"""
    c = script.characters.get(key)
    if not c:
        raise KeyError(f"未定义角色: {key}")
    aspect = script.project.aspect_ratio or "9:16"
    parts = [
        (script.project.style or "cinematic realistic"),
        f"character portrait of {c.name}",
        c.appearance,
        c.outfit,
        "front view, upper body, neutral expression, plain softly-lit background",
        "consistent character reference sheet, photorealistic, high detail",
        f"vertical {aspect}" if aspect == "9:16" else f"aspect {aspect}",
        "no readable text, no watermark",
    ]
    if script.project.negative:
        parts.append(f"避免：{script.project.negative}")
    return "，".join(p for p in parts if p and p.strip())


def resolve_motion_preset(shot: Shot) -> str:
    """由 motion/shot_type 关键词推断 Ken Burns 动效预设（回退或静态镜头用）。"""
    text = f"{shot.motion} {shot.action}".lower()
    for keywords, preset in _MOTION_KEYWORDS:
        if any(k.lower() in text for k in keywords):
            return preset
    return "zoom_in"


def resolution_for(script: Script) -> str:
    return ASPECT_TO_RESOLUTION.get(script.project.aspect_ratio or "9:16", "1080x1920")
