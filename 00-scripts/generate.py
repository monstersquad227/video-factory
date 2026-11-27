#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分镜脚本生成模块（流水线 step 0）。

输入「一段话」（主题/故事），调用阿里云百炼 OpenAI 兼容接口（默认 qwen3.6-plus）把它
改写成 ``script_template.json`` 定义的分镜脚本，输出到 ``projects/<id>/script.json``。

要点：
- LLM 只填**结构化字段**（project / characters / scenes[].shots[] 的 shot_type/characters/
  action/motion/narration/dialogue/duration_sec），**不写 image_prompt/video_prompt**——
  提示词由 ``common.prompts`` 按 prompt_templates 代码拼接，保证角色一致性；
- 用 ``common.schema``（Pydantic）做硬校验（类型/枚举/角色引用），``soft_validate`` 做规则校验
  （每场景≥2 镜头、时长范围、镜头数与目标时长匹配）；不合法则把错误回灌模型自动修复；
- 端点 = ``{DASHSCOPE_BASE_URL}/chat/completions``，``response_format=json_object``，messages 必含 "JSON"。

CLI 示例::

    python 00-scripts/generate.py --text "一个关于坚持与告别的短剧" \
        --project-id the-thousand-afternoons --target-duration 70

    python 00-scripts/generate.py --text "..." --dry-run     # 只看 messages/payload
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import project as project_io  # noqa: E402
from common import schema as schema_mod  # noqa: E402

try:
    import requests
except ImportError:  # pragma: no cover
    sys.stderr.write("缺少依赖 requests，请先执行: pip install -r 00-scripts/requirements.txt\n")
    raise
try:
    from pydantic import ValidationError
except ImportError:  # pragma: no cover
    sys.stderr.write("缺少依赖 pydantic，请先执行: pip install -r common/requirements.txt\n")
    raise
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]


logger = logging.getLogger("scripts.generate")

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
PROJECTS_ROOT = _PROJECT_ROOT / "projects"

DEFAULT_MODEL = "qwen3.6-plus"     # Token Plan 网关实测可用的 plus 档（非思考，JSON 稳定）
DEFAULT_TARGET_DURATION = 70.0
DEFAULT_ASPECT = "9:16"
DEFAULT_LANGUAGE = "zh-CN"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TIMEOUT = 240.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_REPAIR = 2
CONNECT_TIMEOUT = 10.0
CHAT_PATH = "/chat/completions"
RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
# 可选 TTS 音色（qwen-audio-3.0-tts-plus），供 LLM 为角色选 voice
VOICE_HINT = "longanlufeng(男·明亮) / longanlingxin(女·知心温暖) / longanhuan_v3.6 / longanlingxi"


class ScriptGenError(Exception):
    """脚本生成基础异常。"""


class ConfigError(ScriptGenError):
    """配置缺失或非法。"""


class APIError(ScriptGenError):
    """LLM 接口异常。"""


class SchemaError(ScriptGenError):
    """模型输出结构/规则校验失败。"""


@dataclass
class ScriptsConfig:
    api_key: str
    base_url: str
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    max_repair: int = DEFAULT_MAX_REPAIR

    @property
    def chat_endpoint(self) -> str:
        base = (self.base_url or "").strip().rstrip("/")
        if not base:
            raise ConfigError("缺少 DASHSCOPE_BASE_URL")
        return f"{base}{CHAT_PATH}"

    def safe_snapshot(self) -> dict:
        d = asdict(self)
        d.pop("api_key", None)
        return d


@dataclass
class GenParams:
    text: str
    project_id: str
    target_duration: float = DEFAULT_TARGET_DURATION
    aspect: str = DEFAULT_ASPECT
    style: str = ""
    language: str = DEFAULT_LANGUAGE


def _resolve_env_path(explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None
    for c in (_SCRIPT_DIR / ".env", _PROJECT_ROOT / ".env", Path.cwd() / ".env"):
        if c.exists():
            return c
    return None


def load_config(env_path: Optional[Path] = None, **overrides: Any) -> ScriptsConfig:
    resolved = _resolve_env_path(env_path)
    if load_dotenv is not None:
        load_dotenv(resolved, override=False) if resolved else load_dotenv(override=False)
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    base_url = (os.getenv("DASHSCOPE_BASE_URL") or "").strip()
    if not api_key:
        raise ConfigError("缺少 DASHSCOPE_API_KEY，请在 00-scripts/.env 配置")
    if not base_url:
        raise ConfigError("缺少 DASHSCOPE_BASE_URL（OpenAI 兼容网关，如 .../compatible-mode/v1）")

    def _num(name: str, default: Any, cast: type) -> Any:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        try:
            return cast(raw.strip())
        except ValueError as exc:
            raise ConfigError(f"环境变量 {name} 非法: {raw!r}") from exc

    kwargs: dict = {
        "api_key": api_key, "base_url": base_url,
        "model": (os.getenv("SCRIPTS_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "temperature": _num("SCRIPTS_TEMPERATURE", DEFAULT_TEMPERATURE, float),
        "timeout": _num("SCRIPTS_TIMEOUT", DEFAULT_TIMEOUT, float),
        "max_retries": _num("SCRIPTS_MAX_RETRIES", DEFAULT_MAX_RETRIES, int),
        "max_repair": _num("SCRIPTS_MAX_REPAIR", DEFAULT_MAX_REPAIR, int),
    }
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    cfg = ScriptsConfig(**kwargs)
    cfg.chat_endpoint
    return cfg


def slugify(text: str, fallback: str = "episode") -> str:
    slug = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", (text or "").strip()).strip("-")
    return (slug[:40] or fallback)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
def build_system_prompt(params: GenParams) -> str:
    target_shots = max(4, int(round(params.target_duration / 2.7)))
    scene_lo = max(3, int(round(params.target_duration / 9.0)))
    scene_hi = max(scene_lo + 1, int(round(params.target_duration / 5.0)) + 1)
    return f"""你是专业 AI 短剧分镜编剧。用户给你一段主题/故事文字，你要把它改写成用于自动化视频生产的分镜脚本，并严格输出为 JSON（只输出 JSON，不要任何解释，不要 Markdown 代码围栏）。

必须严格遵守如下 JSON 结构（键名用英文；叙述文本用中文）：
{{
  "project": {{
    "id": "{params.project_id}",
    "title": "片名",
    "aspect_ratio": "{params.aspect}",
    "target_duration_sec": {params.target_duration:g},
    "style": "统一画面风格，如：写实电影感，黄昏暖光，浅景深",
    "negative": "文字乱码, 多余手指, 变形, 模糊",
    "language": "{params.language}"
  }},
  "characters": {{
    "角色key_英文小写": {{
      "name": "角色名",
      "appearance": "固定外貌：年龄、性别、发型、体型、肤色、五官特征（全片原样复用，不要改措辞）",
      "outfit": "固定服装与道具",
      "voice": "TTS音色ID"
    }}
  }},
  "scenes": [
    {{
      "id": "s1",
      "name": "场景名",
      "location": "地点",
      "time_of_day": "时间/光线，如：黄昏",
      "shots": [
        {{
          "id": "s1-1",
          "shot_type": "wide|full|medium|close_up|extreme_close_up",
          "characters": ["角色key"],
          "action": "画面中发生了什么（静态首帧，具体可视）",
          "motion": "镜头运动与人物动作（用于图生视频）",
          "narration": "旁白，没有则空字符串",
          "dialogue": [{{"speaker": "角色key", "text": "台词"}}],
          "duration_sec": 3.0
        }}
      ]
    }}
  ],
  "prompt_templates": {{
    "keyframe": "{{style}}，{{shot_type}}，{{location}}，{{time_of_day}}，{{characters_desc}}，{{action}}",
    "video": "{{motion}}，时长{{duration_sec}}秒"
  }}
}}

规则：
1. 只输出上述结构的 JSON。**不要**输出 image_prompt / video_prompt（提示词由代码按 prompt_templates 自动拼接）。prompt_templates 原样照抄上面两行即可。
2. characters：为出现的主要人物建立固定档案；appearance/outfit 一旦确定，全片所有镜头必须原样复用（保证一致性）；voice 从这些可用音色里选：{VOICE_HINT}。旁白（narration）不属于任何角色，用全局旁白音色，无需在 characters 里建旁白角色。
3. 时长：全片总时长≈{params.target_duration:g} 秒；拆成 {scene_lo}~{scene_hi} 个场景；每个场景**必须** 2~4 个镜头（严禁只有 1 个镜头的场景）；单镜头 duration_sec 在 1.5~4 秒；全集镜头总数约 {target_shots} 个。
4. 命名：scene.id 用 s1、s2…；shot.id 用「场景id-序号」，如 s1-1、s1-2。
5. shot_type 只能取枚举值；相邻镜头的景别/构图/运动要不同，避免重复画面。
6. dialogue 的 speaker 必须是 characters 里已定义的 key；一个镜头可只有 narration、只有 dialogue、两者都有、或都为空（纯画面）。
7. action 描述静态画面（用于生成首帧图）；motion 描述随时间发生的动作与镜头运动（用于图生视频）。
8. 内容忠实于用户文案，口语化、有画面感、有情绪与节奏。"""


def build_user_prompt(params: GenParams) -> str:
    return (f"故事/主题：{params.text.strip()}\n\n"
            f"参数：目标总时长 {params.target_duration:g} 秒；画幅 {params.aspect}；"
            f"语言 {params.language}；风格 {params.style or '（由你决定，写实电影感优先）'}。\n"
            f"请据此产出完整分镜脚本 JSON。")


# --------------------------------------------------------------------------- #
# LLM 调用与解析
# --------------------------------------------------------------------------- #
def call_llm(messages: list, cfg: ScriptsConfig) -> tuple[str, dict]:
    url = cfg.chat_endpoint
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    payload = {"model": cfg.model, "messages": messages,
               "temperature": cfg.temperature, "response_format": {"type": "json_object"}}
    attempts = max(1, cfg.max_retries + 1)
    last: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            logger.info("→ 调用 %s（attempt %d/%d, messages=%d）", cfg.model, attempt, attempts, len(messages))
            resp = requests.post(url, headers=headers, json=payload, timeout=(CONNECT_TIMEOUT, cfg.timeout))
        except requests.exceptions.RequestException as exc:
            last = APIError(f"调用 LLM 网络异常: {exc}")
        else:
            if resp.status_code in RETRYABLE_HTTP_STATUS:
                last = APIError(f"LLM 服务暂时不可用: HTTP {resp.status_code} | {(resp.text or '')[:300]}")
            elif resp.status_code >= 400:
                raise APIError(f"LLM 接口返回错误: HTTP {resp.status_code} | {(resp.text or '')[:600]}")
            else:
                try:
                    data = resp.json()
                except ValueError as exc:
                    raise APIError(f"LLM 响应非合法 JSON: {(resp.text or '')[:300]}") from exc
                choices = data.get("choices") or []
                if not choices:
                    raise APIError(f"LLM 响应缺少 choices: {json.dumps(data, ensure_ascii=False)[:400]}")
                content = (choices[0].get("message") or {}).get("content") or ""
                if not content.strip():
                    raise APIError("LLM 返回内容为空")
                return content, (data.get("usage") or {})
        if attempt < attempts:
            delay = min(2.0 * (2 ** (attempt - 1)), 20.0)
            logger.warning("LLM 调用失败，%.1fs 后重试：%s", delay, last)
            time.sleep(delay)
    raise last or APIError("LLM 调用失败")


def parse_json_content(content: str) -> dict:
    text = (content or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        l, r = text.find("{"), text.rfind("}")
        if l != -1 and r != -1 and r > l:
            text = text[l:r + 1]
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise SchemaError(f"模型输出不是合法 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaError("模型输出 JSON 顶层不是对象")
    return data


def validate(data: dict) -> tuple[list, list, Optional[Any]]:
    """Pydantic 硬校验 + soft_validate 规则校验，返回 (errors, warnings, Script|None)。"""
    try:
        script = schema_mod.Script.model_validate(data)
    except ValidationError as exc:
        errs = [f"{'.'.join(str(x) for x in e.get('loc', []))}: {e.get('msg')}" for e in exc.errors()]
        return errs or ["Pydantic 校验失败"], [], None
    errs, warns = schema_mod.soft_validate(script)
    return errs, warns, script


def apply_params(script: Any, params: GenParams) -> Any:
    """用 CLI 参数覆盖 project 的关键字段（保证与命令行一致）。"""
    script.project.id = params.project_id
    script.project.aspect_ratio = params.aspect
    script.project.target_duration_sec = params.target_duration
    script.project.language = params.language
    if params.style:
        script.project.style = params.style
    return script


def generate_script(params: GenParams, cfg: ScriptsConfig, project_dir: Path,
                    dry_run: bool = False) -> dict:
    system_prompt = build_system_prompt(params)
    user_prompt = build_user_prompt(params)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    if dry_run:
        print(json.dumps({"ok": True, "dry_run": True, "endpoint": cfg.chat_endpoint,
                          "model": cfg.model, "project_dir": str(project_dir),
                          "system_prompt": system_prompt, "user_prompt": user_prompt},
                         ensure_ascii=False))
        return {"dry_run": True}

    started = time.perf_counter()
    script = None
    errors: list = []
    warnings: list = []
    usage: dict = {}
    repairs = 0
    for attempt in range(cfg.max_repair + 1):
        content, usage = call_llm(messages, cfg)
        try:
            data = parse_json_content(content)
        except SchemaError as exc:
            errors, script = [str(exc)], None
        else:
            errors, warnings, script = validate(data)
        if not errors and script is not None:
            break
        repairs += 1
        if attempt >= cfg.max_repair:
            break
        logger.warning("脚本校验未通过（%d 处），回灌模型修复（第 %d/%d 次）", len(errors), repairs, cfg.max_repair)
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content":
                         "上面的 JSON 存在以下问题，请修正后重新输出**完整**的 JSON（只输出 JSON）：\n- "
                         + "\n- ".join(errors[:20])})
    if script is None or errors:
        raise SchemaError("模型输出多次校验仍不合法：\n- " + "\n- ".join(errors[:20] or ["未知错误"]))

    script = apply_params(script, params)
    project_dir.mkdir(parents=True, exist_ok=True)
    script_path = project_dir / "script.json"
    project_io.save_script(script, script_path)
    for w in warnings:
        logger.warning("  [warn] %s", w)
    return {"script": script, "script_path": script_path, "warnings": warnings,
            "repairs": repairs, "usage": usage, "elapsed": round(time.perf_counter() - started, 2)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts.generate",
        description="一段话 → projects/<id>/script.json（模板 schema，qwen3.6-plus + Pydantic 校验/修复）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="主题/故事（一段话）")
    src.add_argument("--text-file", help="从文件读取主题/故事")
    p.add_argument("--project", help="项目目录（默认 projects/<project-id>）")
    p.add_argument("--project-id", help="项目 id（默认由文案 slug 化）")
    p.add_argument("--target-duration", type=float, default=DEFAULT_TARGET_DURATION, help="目标总时长（秒）")
    p.add_argument("--aspect", default=DEFAULT_ASPECT, help="画幅，如 9:16")
    p.add_argument("--style", default="", help="统一画面风格提示")
    p.add_argument("--language", default=DEFAULT_LANGUAGE, help="语言，如 zh-CN")
    p.add_argument("--model", help=f"LLM 模型（默认 {DEFAULT_MODEL}）")
    p.add_argument("--temperature", type=float, help="采样温度")
    p.add_argument("--max-repair", type=int, help="校验失败时的最大修复重试次数")
    p.add_argument("--env", help="指定 .env 路径")
    p.add_argument("--dry-run", action="store_true", help="只打印 messages/payload，不调用")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)
    try:
        if args.text:
            text = args.text
        else:
            tf = Path(args.text_file).expanduser()
            if not tf.exists():
                raise ScriptGenError(f"文案文件不存在: {tf}")
            text = tf.read_text(encoding="utf-8").strip()
        if not text.strip():
            raise ScriptGenError("文案为空")

        cfg = load_config(env_path=Path(args.env) if args.env else None, model=args.model,
                          temperature=args.temperature, max_repair=args.max_repair)
        project_id = args.project_id or slugify(text)
        project_dir = Path(args.project).expanduser() if args.project else (PROJECTS_ROOT / project_id)
        params = GenParams(text=text, project_id=project_id, target_duration=args.target_duration,
                           aspect=args.aspect, style=args.style, language=args.language)

        result = generate_script(params, cfg, project_dir, dry_run=args.dry_run)
        if args.dry_run:
            return 0

        script = result["script"]
        n_scenes = len(script.scenes)
        n_shots = len(script.all_shots())
        n_chars = len(script.characters)
        nominal = round(sum(s.duration_sec for s in script.all_shots()), 2)
        logger.info("— 结果 —")
        logger.info("  模型      : %s", cfg.model)
        logger.info("  角色数    : %d（%s）", n_chars, ", ".join(script.characters.keys()))
        logger.info("  场景/镜头 : %d / %d（名义总时长 %ss，目标 %ss）",
                    n_scenes, n_shots, nominal, script.project.target_duration_sec)
        logger.info("  修复次数  : %d", result["repairs"])
        logger.info("  耗时      : %.1fs", result["elapsed"])
        logger.info("  脚本文件  : %s", Path(result["script_path"]).resolve())
        print(json.dumps({
            "ok": True, "project_dir": str(Path(project_dir).resolve()),
            "script_path": str(Path(result["script_path"]).resolve()),
            "project_id": project_id, "scene_count": n_scenes, "shot_count": n_shots,
            "character_count": n_chars, "nominal_duration": nominal,
            "repairs": result["repairs"], "usage": result["usage"], "warnings": result["warnings"],
        }, ensure_ascii=False))
        return 0
    except ConfigError as exc:
        logger.error("配置错误: %s", exc)
        print(json.dumps({"ok": False, "error_type": "ConfigError", "error": str(exc)}, ensure_ascii=False))
        return 3
    except SchemaError as exc:
        logger.error("脚本校验失败: %s", exc)
        print(json.dumps({"ok": False, "error_type": "SchemaError", "error": str(exc)}, ensure_ascii=False))
        return 4
    except ScriptGenError as exc:
        logger.error("脚本生成失败: %s", exc)
        print(json.dumps({"ok": False, "error_type": "ScriptGenError", "error": str(exc)}, ensure_ascii=False))
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
