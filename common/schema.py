#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""script.json 数据模型（严格对应 script_template.json），Pydantic v2。

script.json 是整条短剧流水线的唯一数据源：project / characters / scenes[].shots[]
/ prompt_templates，以及每个 shot 的 status 与 outputs（由各模块回写，实现续跑）。

字段与模板一致；额外增加 3 个运行期字段（可选，模板没有但流水线需要）：
- ``runtime_sec``：01-voice 测得的 shot 真实音频时长（无音频的纯视觉 shot 为空）；
- ``start`` / ``end``：01-voice 计算的绝对时间轴。
"""

from __future__ import annotations

from typing import Iterator, Literal, Optional

from pydantic import BaseModel, Field, model_validator

ShotType = Literal["wide", "full", "medium", "close_up", "extreme_close_up"]
ShotStatusValue = Literal["pending", "running", "done", "failed", "skipped"]

SHOT_TYPES = ("wide", "full", "medium", "close_up", "extreme_close_up")
ASPECT_RATIOS = ("9:16", "16:9", "1:1", "3:4", "4:3", "21:9")

MIN_SHOT = 1.5
MAX_SHOT = 4.0

ASPECT_TO_RESOLUTION = {
    "9:16": "1080x1920", "16:9": "1920x1080", "1:1": "1080x1080",
    "3:4": "1080x1440", "4:3": "1440x1080", "21:9": "1920x823",
}


class DialogueLine(BaseModel):
    """一条对白：speaker 必须是 characters 的 key。"""

    speaker: str
    text: str


class ShotStatus(BaseModel):
    keyframe: ShotStatusValue = "pending"
    video: ShotStatusValue = "pending"
    audio: ShotStatusValue = "pending"


class ShotOutputs(BaseModel):
    keyframe: Optional[str] = None
    video: Optional[str] = None
    audio: Optional[str] = None


class Shot(BaseModel):
    id: str
    shot_type: ShotType = "medium"
    characters: list[str] = Field(default_factory=list)
    action: str = ""            # 画面中发生了什么（用于生成首帧图）
    motion: str = ""            # 镜头运动与人物动作（用于图生视频）
    narration: str = ""         # 旁白，没有则空串
    dialogue: list[DialogueLine] = Field(default_factory=list)
    duration_sec: float = 3.0   # 名义时长
    # 运行期字段（01-voice 回写）
    runtime_sec: Optional[float] = None
    start: Optional[float] = None
    end: Optional[float] = None
    status: ShotStatus = Field(default_factory=ShotStatus)
    outputs: ShotOutputs = Field(default_factory=ShotOutputs)


class Character(BaseModel):
    name: str
    appearance: str = ""        # 固定外貌，全片原样复用
    outfit: str = ""            # 固定服装与道具
    voice: str = ""             # TTS 音色 ID
    ref_image: Optional[str] = None


class Scene(BaseModel):
    id: str
    name: str = ""
    location: str = ""
    time_of_day: str = ""
    shots: list[Shot] = Field(default_factory=list)


class Project(BaseModel):
    id: str
    title: str = ""
    aspect_ratio: str = "9:16"
    target_duration_sec: float = 70
    style: str = ""
    negative: str = ""
    language: str = "zh-CN"


class PromptTemplates(BaseModel):
    keyframe: str = "{style}，{shot_type}，{location}，{time_of_day}，{characters_desc}，{action}"
    video: str = "{motion}，时长{duration_sec}秒"


class Script(BaseModel):
    """整份脚本（script.json 根对象）。"""

    project: Project
    characters: dict[str, Character] = Field(default_factory=dict)
    scenes: list[Scene] = Field(default_factory=list)
    prompt_templates: PromptTemplates = Field(default_factory=PromptTemplates)

    # ---- 便捷方法 ----
    def iter_shots(self) -> Iterator[tuple[Scene, Shot]]:
        for scene in self.scenes:
            for shot in scene.shots:
                yield scene, shot

    def all_shots(self) -> list[Shot]:
        return [shot for _, shot in self.iter_shots()]

    def get_shot(self, shot_id: str) -> Optional[Shot]:
        for _, shot in self.iter_shots():
            if shot.id == shot_id:
                return shot
        return None

    def character(self, key: str) -> Optional[Character]:
        return self.characters.get(key)

    @model_validator(mode="after")
    def _check_refs(self) -> "Script":
        """校验 shot.characters 与 dialogue.speaker 的 key 都存在于 characters。"""
        known = set(self.characters.keys())
        for scene in self.scenes:
            for shot in scene.shots:
                for key in shot.characters:
                    if key not in known:
                        raise ValueError(
                            f"shot {shot.id} 引用了未定义的角色 key: {key!r}（characters 里没有）")
                for line in shot.dialogue:
                    if line.speaker not in known:
                        raise ValueError(
                            f"shot {shot.id} 对白 speaker 未定义: {line.speaker!r}（characters 里没有）")
        return self


def soft_validate(script: Script) -> tuple[list[str], list[str]]:
    """规则性软校验（供 00-scripts 触发修复），返回 (errors, warnings)。

    硬结构（类型/枚举/角色引用）由 Pydantic 保证；这里检查叙事规则：
    每场景≥2 镜头、时长范围、总镜头数与目标时长匹配、video 预算等。
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not script.scenes:
        errors.append("scenes 为空")
        return errors, warnings

    total_shots = 0
    total_nominal = 0.0
    for scene in script.scenes:
        if len(scene.shots) < 2:
            errors.append(f"scene {scene.id} 只有 {len(scene.shots)} 个镜头；每个场景至少 2 个")
        for shot in scene.shots:
            total_shots += 1
            total_nominal += shot.duration_sec
            if not (MIN_SHOT - 1e-6 <= shot.duration_sec <= MAX_SHOT + 1e-6):
                warnings.append(
                    f"shot {shot.id} duration_sec={shot.duration_sec} 超出 [{MIN_SHOT},{MAX_SHOT}]")
            if not shot.action.strip():
                errors.append(f"shot {shot.id} 缺少 action（用于生成首帧）")

    target = script.project.target_duration_sec
    if target > 0:
        expect = max(4, int(round(target / 2.7)))
        if not (int(expect * 0.6) <= total_shots <= int(expect * 1.6) + 2):
            warnings.append(f"镜头总数 {total_shots} 偏离目标约 {expect}（target={target}s）")
        if abs(total_nominal - target) > max(6.0, target * 0.25):
            warnings.append(f"名义总时长 {total_nominal:.1f}s 与 target {target}s 偏差较大")
    return errors, warnings
