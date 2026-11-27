#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""video-factory 公共库：script.json 数据模型、提示词模板拼接、项目 IO 与状态回写。

各模块（00-scripts … 05-compose、run.py）通过把仓库根加入 sys.path 后 ``import common.*`` 复用。
"""

from . import project, prompts, schema
from .schema import (
    ASPECT_TO_RESOLUTION,
    Character,
    DialogueLine,
    Project,
    PromptTemplates,
    Scene,
    Script,
    Shot,
    ShotOutputs,
    ShotStatus,
    soft_validate,
)

__all__ = [
    "schema", "prompts", "project",
    "Script", "Project", "Character", "Scene", "Shot",
    "DialogueLine", "ShotStatus", "ShotOutputs", "PromptTemplates",
    "soft_validate", "ASPECT_TO_RESOLUTION",
]
