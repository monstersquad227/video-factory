#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阿里云百炼 wan2.7-image-pro 图片生成服务模块。

功能：
- 通过阿里云百炼官方 HTTP 异步接口调用 ``wan2.7-image-pro`` 生成图片；
- API Key、接入点等配置从 ``.env`` 读取，禁止硬编码；
- 支持直接传入提示词，或读取 JSON 脚本中的 ``visual.prompt`` 批量生成分镜图片；
- 支持 ``1K`` / ``2K`` / ``4K`` 分辨率档位，也支持 ``WIDTHxHEIGHT`` 自定义尺寸；
- 输出 PNG，并自动下载模型返回的临时图片 URL；
- 处理 API 异常、HTTP 超时、任务失败、下载失败，并进行有限重试；
- 输出每张图片的生成耗时、路径、实际分辨率和文件大小。

CLI 示例::

    python 02-image/generate.py --prompt "电影感城市夜景，竖屏 9:16" --size 1080x1920
    python 02-image/generate.py --script LossAversion/Script.json
    python 02-image/generate.py --script LossAversion/Script.json --scene 1,3,5 --size 2K

作为模块调用::

    from generate import load_config, generate_image
    cfg = load_config()
    result = generate_image("电影感城市夜景", cfg)
    print(result.image_path)
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import struct
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import project as project_io  # noqa: E402
from common import prompts as prompt_kit  # noqa: E402
from common import schema as schema_mod  # noqa: E402

try:
    import requests
except ImportError:  # pragma: no cover
    sys.stderr.write("缺少依赖 requests，请先执行: pip install -r 02-image/requirements.txt\n")
    raise

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]


logger = logging.getLogger("image.generate")

MODEL_NAME = "wan2.7-image-pro"
SUPPORTED_MODELS = ("wan2.7-image-pro", "wan2.7-image")
DEFAULT_SIZE = "2K"
DEFAULT_BASE_ORIGIN = "https://dashscope.aliyuncs.com"
IMAGE_GENERATION_PATH = "/api/v1/services/aigc/image-generation/generation"
TASK_PATH_PREFIX = "/api/v1/tasks/"
CONNECT_TIMEOUT = 10.0
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_PROMPT_CHARS = 5000
MIN_TEXT_TO_IMAGE_PIXELS = 768 * 768
MAX_PRO_TEXT_TO_IMAGE_PIXELS = 4096 * 4096
MAX_STANDARD_TEXT_TO_IMAGE_PIXELS = 2048 * 2048
MAX_OUTPUT_SIDE = 8000
MAX_ASPECT_RATIO = 8.0
MAX_RETRY_DELAY = 30.0
MAX_SEED = 2147483647
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

RESOLUTION_PRESETS = {
    "1K": 1024 * 1024,
    "2K": 2048 * 2048,
    "4K": 4096 * 4096,
}
RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
NON_RETRYABLE_ERROR_CODES = {
    "AccessDenied",
    "Arrearage",
    "BadRequest",
    "DataInspectionFailed",
    "Forbidden",
    "InvalidApiKey",
    "InvalidParameter",
    "InvalidWorkspace",
    "ModelNotFound",
    "UnsupportedOperation",
}
NON_RETRYABLE_MESSAGE_KEYWORDS = (
    "access denied",
    "arrearage",
    "content policy",
    "data inspection",
    "forbidden",
    "invalid api key",
    "invalid parameter",
    "invalidapikey",
    "invalidparameter",
    "moderation",
    "nsfw",
    "unsupported",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = _SCRIPT_DIR / "output"

_REGION_HOSTS = {
    "cn-beijing": "{ws}.cn-beijing.maas.aliyuncs.com",
    "ap-southeast-1": "{ws}.ap-southeast-1.maas.aliyuncs.com",
}


class ImageGenerationError(Exception):
    """图片生成服务基础异常。"""


class ConfigError(ImageGenerationError):
    """配置缺失或非法。"""


class ScriptError(ImageGenerationError):
    """JSON 脚本解析失败。"""


class APIError(ImageGenerationError):
    """阿里云百炼 HTTP API 异常。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
        request_id: Optional[str] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.retryable = retryable


class TaskError(ImageGenerationError):
    """生图任务进入失败或未知状态。"""

    def __init__(
        self,
        message: str,
        *,
        task_id: str = "",
        task_status: str = "",
        code: Optional[str] = None,
        request_id: Optional[str] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.task_id = task_id
        self.task_status = task_status
        self.code = code
        self.request_id = request_id
        self.retryable = retryable


class TaskTimeoutError(TaskError):
    """生图任务轮询超时。"""

    def __init__(
        self,
        message: str,
        *,
        task_id: str = "",
        task_status: str = "",
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            task_id=task_id,
            task_status=task_status,
            request_id=request_id,
            retryable=False,
        )


class DownloadError(ImageGenerationError):
    """图片下载或写盘异常。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass
class ImageConfig:
    """图片生成配置。敏感字段不会出现在快照中。"""

    api_key: str
    base_url: Optional[str] = None
    workspace_id: str = ""
    region: str = "cn-beijing"
    model: str = MODEL_NAME
    size: str = DEFAULT_SIZE
    watermark: bool = False
    thinking_mode: bool = True
    seed: Optional[int] = None
    request_timeout: float = 120.0
    task_timeout: float = 600.0
    poll_interval: float = 5.0
    download_timeout: float = 120.0
    max_retries: int = 2
    retry_backoff: float = 2.0

    @property
    def max_attempts(self) -> int:
        return max(1, self.max_retries + 1)

    @property
    def http_timeout(self) -> tuple:
        return (CONNECT_TIMEOUT, self.request_timeout)

    @property
    def api_origin(self) -> str:
        if self.base_url:
            parsed = urlparse(self.base_url.strip())
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ConfigError(f"DASHSCOPE_BASE_URL 非法: {self.base_url!r}")
            return f"{parsed.scheme}://{parsed.netloc}"
        if self.workspace_id:
            template = _REGION_HOSTS.get(self.region)
            if not template:
                raise ConfigError(f"不支持的 region: {self.region!r}，可选: {sorted(_REGION_HOSTS)}")
            return f"https://{template.format(ws=self.workspace_id)}"
        return DEFAULT_BASE_ORIGIN

    @property
    def create_endpoint(self) -> str:
        return f"{self.api_origin}{IMAGE_GENERATION_PATH}"

    def task_endpoint(self, task_id: str) -> str:
        if not task_id:
            raise APIError("缺少 task_id，无法查询任务")
        return f"{self.api_origin}{TASK_PATH_PREFIX}{quote(str(task_id), safe='')}"

    def safe_snapshot(self) -> dict:
        data = asdict(self)
        data.pop("api_key", None)
        if data.get("workspace_id"):
            data["workspace_id"] = _mask(data["workspace_id"])
        data["api_origin"] = self.api_origin
        return data


@dataclass
class ImageResult:
    """一张图片的生成结果。"""

    image_path: Path
    requested_size: str
    width: int
    height: int
    actual_size: str
    file_size: int
    duration_seconds: float
    request_id: str
    task_id: str
    image_url: str
    model: str = MODEL_NAME
    scene_id: Optional[str] = None
    reported_size: str = ""
    config_snapshot: dict = field(default_factory=dict)


def _mask(value: str, keep: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= keep:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * 6}{value[-2:]}"


def _human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{int(num)} B" if unit == "B" else f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} TB"


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    index = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def _parse_bool(raw: Any, name: str) -> bool:
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"环境变量 {name} 不是合法布尔值: {raw!r}")


def _env_optional(name: str) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return _parse_bool(raw, name)


def _env_number(name: str, default: Any, cast: type) -> Any:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"环境变量 {name} 不是合法数字: {raw!r}") from exc


def normalize_size(value: Any, model: str = MODEL_NAME) -> str:
    """将 ``1K``、``2K``、``4K`` 或 ``WIDTHxHEIGHT`` 标准化为 API 接受的字符串。"""
    if value is None or str(value).strip() == "":
        return DEFAULT_SIZE

    text = str(value).strip()
    preset = text.upper()
    if preset in RESOLUTION_PRESETS:
        if preset == "4K" and model != "wan2.7-image-pro":
            raise ConfigError("4K 仅 wan2.7-image-pro 的文生图场景支持")
        return preset

    match = re.fullmatch(r"(\d{1,5})\s*[xX*×:]\s*(\d{1,5})", text)
    if not match:
        raise ConfigError(
            f"不支持的图片尺寸: {value!r}；可选 1K/2K/4K，或 WIDTHxHEIGHT / WIDTH*HEIGHT"
        )

    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ConfigError(f"图片宽高必须为正整数，当前: {value!r}")
    if width > MAX_OUTPUT_SIDE or height > MAX_OUTPUT_SIDE:
        raise ConfigError(f"图片宽高不能超过 {MAX_OUTPUT_SIDE} 像素，当前: {width}x{height}")

    total_pixels = width * height
    max_pixels = (
        MAX_PRO_TEXT_TO_IMAGE_PIXELS
        if model == "wan2.7-image-pro"
        else MAX_STANDARD_TEXT_TO_IMAGE_PIXELS
    )
    if not MIN_TEXT_TO_IMAGE_PIXELS <= total_pixels <= max_pixels:
        raise ConfigError(
            f"{model} 文生图总像素需在 [{MIN_TEXT_TO_IMAGE_PIXELS}, {max_pixels}] 之间，"
            f"当前 {width}x{height}={total_pixels}"
        )

    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > MAX_ASPECT_RATIO:
        raise ConfigError(f"图片宽高比需在 [1:8, 8:1] 之间，当前: {width}x{height}")

    return f"{width}*{height}"


def _filename_stem(filename: Any) -> str:
    stem = Path(str(filename or "")).stem
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-._")
    return slug or "image"


def _scene_file_stem(scene_id: Any, scene_index: Optional[int] = None) -> str:
    raw = str(scene_id or "").strip()
    if raw.isdigit():
        return f"scene_{int(raw):03d}"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-._")
    if slug:
        return f"scene_{slug}"
    if scene_index is not None:
        return f"scene_{scene_index:03d}"
    return "scene"


def _default_output_dir(output_root: Optional[Path] = None) -> Path:
    root = Path(output_root).expanduser() if output_root else DEFAULT_OUTPUT_ROOT
    return root / datetime.now().strftime("%Y%m%d_%H%M%S")


def _prepare_destination(
    output_path: Optional[Path] = None,
    output_root: Optional[Path] = None,
    scene_id: Optional[str] = None,
    filename: Optional[str] = None,
    scene_index: Optional[int] = None,
) -> Path:
    if output_path:
        destination = Path(output_path).expanduser()
        if destination.suffix.lower() != ".png":
            destination = destination.with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return _unique_path(destination.parent, destination.stem, ".png")

    directory = Path(output_root).expanduser() if output_root else _default_output_dir()
    directory.mkdir(parents=True, exist_ok=True)
    if filename:
        stem = _filename_stem(filename)
    elif scene_id is not None:
        stem = _scene_file_stem(scene_id, scene_index)
    else:
        stem = "image"
    return _unique_path(directory, stem, ".png")


def _resolve_env_path(explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.exists() else None
    candidates = [
        _SCRIPT_DIR / ".env",
        _SCRIPT_DIR.parent / ".env",
        Path.cwd() / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _validate_config(cfg: ImageConfig) -> None:
    if not cfg.api_key:
        raise ConfigError("缺少 DASHSCOPE_API_KEY")
    if cfg.model not in SUPPORTED_MODELS:
        raise ConfigError(f"不支持的模型: {cfg.model!r}，可选: {SUPPORTED_MODELS}")
    if cfg.request_timeout <= 0:
        raise ConfigError(f"IMAGE_REQUEST_TIMEOUT 必须大于 0，当前 {cfg.request_timeout}")
    if cfg.task_timeout <= 0:
        raise ConfigError(f"IMAGE_TASK_TIMEOUT 必须大于 0，当前 {cfg.task_timeout}")
    if cfg.poll_interval <= 0:
        raise ConfigError(f"IMAGE_POLL_INTERVAL 必须大于 0，当前 {cfg.poll_interval}")
    if cfg.download_timeout <= 0:
        raise ConfigError(f"IMAGE_DOWNLOAD_TIMEOUT 必须大于 0，当前 {cfg.download_timeout}")
    if cfg.max_retries < 0:
        raise ConfigError(f"IMAGE_MAX_RETRIES 不能为负数，当前 {cfg.max_retries}")
    if cfg.retry_backoff < 0:
        raise ConfigError(f"IMAGE_RETRY_BACKOFF 不能为负数，当前 {cfg.retry_backoff}")
    if cfg.seed is not None and not 0 <= cfg.seed <= MAX_SEED:
        raise ConfigError(f"seed 需在 [0, {MAX_SEED}]，当前 {cfg.seed}")
    cfg.size = normalize_size(cfg.size, cfg.model)
    cfg.api_origin


def load_config(env_path: Optional[Path] = None, **overrides: Any) -> ImageConfig:
    """从 ``.env`` 加载配置，并用 ``overrides`` 中非 None 的值覆盖。

    环境变量：
        DASHSCOPE_API_KEY        必填，百炼 API Key
        DASHSCOPE_BASE_URL       可选，显式接入点；会保留主机名并拼接原生图片生成路径
        DASHSCOPE_WORKSPACE_ID   可选，业务空间 ID，用于拼接专属域名
        DASHSCOPE_REGION         可选，cn-beijing（默认）/ ap-southeast-1
        IMAGE_MODEL / IMAGE_SIZE / IMAGE_WATERMARK / IMAGE_THINKING_MODE / IMAGE_SEED
        IMAGE_REQUEST_TIMEOUT / IMAGE_TASK_TIMEOUT / IMAGE_POLL_INTERVAL
        IMAGE_DOWNLOAD_TIMEOUT / IMAGE_MAX_RETRIES / IMAGE_RETRY_BACKOFF
    """
    resolved = _resolve_env_path(env_path)
    if load_dotenv is not None:
        if resolved:
            load_dotenv(resolved, override=False)
            logger.debug("已加载 .env: %s", resolved)
        else:
            load_dotenv(override=False)
    elif resolved:
        logger.warning("未安装 python-dotenv，将仅使用进程环境变量（忽略 %s）", resolved)

    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        raise ConfigError("缺少 DASHSCOPE_API_KEY，请在 .env 中配置（参考 02-image/.env.example）")

    kwargs: dict = {
        "api_key": api_key,
        "base_url": _env_optional("DASHSCOPE_BASE_URL"),
        "workspace_id": (os.getenv("DASHSCOPE_WORKSPACE_ID") or "").strip(),
        "region": (os.getenv("DASHSCOPE_REGION") or "cn-beijing").strip() or "cn-beijing",
        "model": (os.getenv("IMAGE_MODEL") or MODEL_NAME).strip() or MODEL_NAME,
        "size": (os.getenv("IMAGE_SIZE") or DEFAULT_SIZE).strip() or DEFAULT_SIZE,
        "watermark": _env_bool("IMAGE_WATERMARK", False),
        "thinking_mode": _env_bool("IMAGE_THINKING_MODE", True),
        "request_timeout": _env_number("IMAGE_REQUEST_TIMEOUT", 120.0, float),
        "task_timeout": _env_number("IMAGE_TASK_TIMEOUT", 600.0, float),
        "poll_interval": _env_number("IMAGE_POLL_INTERVAL", 5.0, float),
        "download_timeout": _env_number("IMAGE_DOWNLOAD_TIMEOUT", 120.0, float),
        "max_retries": _env_number("IMAGE_MAX_RETRIES", 2, int),
        "retry_backoff": _env_number("IMAGE_RETRY_BACKOFF", 2.0, float),
    }

    seed_raw = os.getenv("IMAGE_SEED")
    if seed_raw and seed_raw.strip():
        kwargs["seed"] = int(seed_raw.strip())

    kwargs.update({key: value for key, value in overrides.items() if value is not None})
    cfg = ImageConfig(**kwargs)
    _validate_config(cfg)
    return cfg


def build_payload(prompt: str, cfg: ImageConfig, ref_images: Optional[list] = None) -> dict:
    """构造 wan2.7-image 请求体。ref_images 非空时走多图参考（content 混合 image+text）。"""
    text = (prompt or "").strip()
    if not text:
        raise ImageGenerationError("图片提示词为空")
    if len(text) > MAX_PROMPT_CHARS:
        logger.warning("提示词超过 %d 字符，将截断后提交", MAX_PROMPT_CHARS)
        text = text[:MAX_PROMPT_CHARS]

    content: list = []
    for uri in (ref_images or []):
        content.append({"image": uri})
    content.append({"text": text})

    parameters: dict = {
        "size": normalize_size(cfg.size, cfg.model),
        "n": 1,
        "watermark": bool(cfg.watermark),
    }
    if not ref_images:
        parameters["thinking_mode"] = bool(cfg.thinking_mode)  # 仅无图输入时生效
    if cfg.seed is not None:
        parameters["seed"] = cfg.seed

    return {
        "model": cfg.model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": parameters,
    }


def _is_retryable_error_code(code: Optional[str], message: Optional[str]) -> bool:
    if code and code in NON_RETRYABLE_ERROR_CODES:
        return False
    text = f"{code or ''} {message or ''}".lower()
    return not any(keyword in text for keyword in NON_RETRYABLE_MESSAGE_KEYWORDS)


def _response_error_parts(response: "requests.Response") -> tuple:
    text = (response.text or "")[:800]
    code = message = request_id = None
    try:
        data = response.json()
        if isinstance(data, dict):
            output = data.get("output") if isinstance(data.get("output"), dict) else {}
            code = data.get("code") or data.get("Code") or output.get("code")
            message = data.get("message") or data.get("Message") or output.get("message")
            request_id = data.get("request_id") or data.get("RequestId")
    except ValueError:
        pass

    if code or message:
        detail = f"HTTP {response.status_code} | code={code} | message={message} | request_id={request_id}"
    else:
        detail = f"HTTP {response.status_code} | {text}"
    return code, message, request_id, detail


def _retry_after_seconds(headers: Any) -> Optional[float]:
    raw = headers.get("Retry-After") if hasattr(headers, "get") else None
    if not raw:
        return None
    try:
        return min(max(float(raw), 0.0), MAX_RETRY_DELAY)
    except (TypeError, ValueError):
        return None


def _retry_delay(attempt: int, cfg: ImageConfig, retry_after: Optional[float] = None) -> float:
    if retry_after is not None:
        return retry_after
    return min(cfg.retry_backoff * (2 ** (attempt - 1)), MAX_RETRY_DELAY)


def _sleep_before_retry(
    attempt: int,
    cfg: ImageConfig,
    operation: str,
    error: Optional[Exception],
    retry_after: Optional[float] = None,
) -> None:
    delay = _retry_delay(attempt, cfg, retry_after)
    logger.warning(
        "%s失败，%.1fs 后重试（第 %d/%d 次）: %s",
        operation,
        delay,
        attempt,
        cfg.max_retries,
        error,
    )
    time.sleep(delay)


def _send_json_request(
    method: str,
    url: str,
    cfg: ImageConfig,
    operation: str,
    headers: Optional[dict] = None,
    payload: Optional[dict] = None,
    timeout: Optional[tuple] = None,
    raise_on_body_code: bool = True,
) -> dict:
    request_timeout = timeout or cfg.http_timeout
    last_error: Optional[APIError] = None

    for attempt in range(1, cfg.max_attempts + 1):
        retry_after: Optional[float] = None
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                json=payload,
                timeout=request_timeout,
            )
        except requests.exceptions.Timeout as exc:
            last_error = APIError(
                f"{operation}超时（>{request_timeout[1]}s）: {exc}",
                retryable=True,
            )
        except requests.exceptions.RequestException as exc:
            last_error = APIError(f"{operation}网络异常: {exc}", retryable=True)
        else:
            if response.status_code in RETRYABLE_HTTP_STATUS:
                code, _, request_id, detail = _response_error_parts(response)
                retry_after = _retry_after_seconds(response.headers)
                last_error = APIError(
                    f"{operation}服务暂时不可用: {detail}",
                    status_code=response.status_code,
                    code=code,
                    request_id=request_id,
                    retryable=True,
                )
            elif response.status_code >= 400:
                code, _, request_id, detail = _response_error_parts(response)
                raise APIError(
                    f"{operation}接口返回错误: {detail}",
                    status_code=response.status_code,
                    code=code,
                    request_id=request_id,
                    retryable=False,
                )
            else:
                try:
                    data = response.json()
                except ValueError as exc:
                    raise APIError(
                        f"{operation}响应不是合法 JSON: {(response.text or '')[:300]}"
                    ) from exc
                if not isinstance(data, dict):
                    raise APIError(f"{operation}响应格式非法: {str(data)[:300]}")

                code = data.get("code") or data.get("Code")
                if raise_on_body_code and code:
                    message = data.get("message") or data.get("Message")
                    request_id = data.get("request_id") or data.get("RequestId")
                    retryable = _is_retryable_error_code(str(code), str(message or ""))
                    error = APIError(
                        f"{operation}返回业务错误: code={code} | message={message} | request_id={request_id}",
                        status_code=response.status_code,
                        code=str(code),
                        request_id=str(request_id or ""),
                        retryable=retryable,
                    )
                    if retryable:
                        last_error = error
                    else:
                        raise error
                return data

        if attempt < cfg.max_attempts:
            _sleep_before_retry(attempt, cfg, operation, last_error, retry_after)

    logger.error("%s重试 %d 次后仍失败", operation, cfg.max_retries)
    raise last_error or APIError(f"{operation}失败: 未知错误")


def _create_task(payload: dict, cfg: ImageConfig) -> tuple:
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    data = _send_json_request(
        "POST",
        cfg.create_endpoint,
        cfg,
        "创建生图任务",
        headers=headers,
        payload=payload,
    )
    output = data.get("output") if isinstance(data.get("output"), dict) else {}
    task_id = output.get("task_id") or data.get("task_id")
    request_id = data.get("request_id") or ""
    if not task_id:
        raise APIError(
            f"创建任务响应缺少 output.task_id: {json.dumps(data, ensure_ascii=False)[:500]}"
        )
    return str(task_id), str(request_id)


def _poll_task(task_id: str, cfg: ImageConfig) -> dict:
    url = cfg.task_endpoint(task_id)
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    deadline = time.monotonic() + cfg.task_timeout

    while True:
        data = _send_json_request(
            "GET",
            url,
            cfg,
            "查询生图任务",
            headers=headers,
            raise_on_body_code=False,
        )
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        status = str(output.get("task_status") or data.get("task_status") or "UNKNOWN").upper()
        request_id = str(data.get("request_id") or "")

        if status in {"PENDING", "RUNNING"}:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TaskTimeoutError(
                    f"生图任务轮询超时（>{cfg.task_timeout}s），task_id={task_id}",
                    task_id=task_id,
                    task_status=status,
                    request_id=request_id,
                )
            logger.debug("任务 %s 状态 %s，%.1fs 后继续轮询", task_id, status, remaining)
            time.sleep(min(cfg.poll_interval, remaining))
            continue

        if status == "SUCCEEDED":
            return data

        code = data.get("code") or output.get("code")
        message = data.get("message") or output.get("message") or output.get("status_message")
        retryable = status != "CANCELED" and _is_retryable_error_code(
            str(code or ""), str(message or "")
        )
        raise TaskError(
            f"生图任务失败: task_id={task_id} | status={status} | code={code} | message={message}",
            task_id=task_id,
            task_status=status,
            code=str(code or "") or None,
            request_id=request_id,
            retryable=retryable,
        )


def _extract_image_result(data: dict) -> tuple:
    output = data.get("output") if isinstance(data.get("output"), dict) else {}
    image_urls = []

    for choice in output.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        for item in message.get("content") or []:
            if isinstance(item, dict) and item.get("image"):
                image_urls.append(str(item["image"]))

    if not image_urls:
        for item in output.get("results") or []:
            if isinstance(item, dict):
                url = item.get("image") or item.get("url")
                if url:
                    image_urls.append(str(url))

    if not image_urls:
        raise APIError(
            f"任务成功但未返回图片 URL: {json.dumps(data, ensure_ascii=False)[:500]}"
        )

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    reported_size = str(usage.get("size") or "")
    request_id = str(data.get("request_id") or "")
    return image_urls[0], reported_size, request_id


def _read_png_info(path: Path) -> tuple:
    width = height = 0
    with path.open("rb") as handle:
        if handle.read(8) != PNG_SIGNATURE:
            raise DownloadError(f"下载内容不是 PNG 文件: {path.name}")

        found_iend = False
        while True:
            raw_length = handle.read(4)
            if len(raw_length) < 4:
                break
            chunk_type = handle.read(4)
            if len(chunk_type) < 4:
                break
            length = struct.unpack(">I", raw_length)[0]

            if chunk_type == b"IHDR":
                if length < 8:
                    raise DownloadError("PNG IHDR 块长度非法")
                data = handle.read(8)
                if len(data) < 8:
                    break
                width, height = struct.unpack(">II", data)
                handle.seek(length - 8, os.SEEK_CUR)
            else:
                handle.seek(length, os.SEEK_CUR)

            crc = handle.read(4)
            if len(crc) < 4:
                break
            if chunk_type == b"IEND":
                found_iend = True
                break

    if not found_iend or width <= 0 or height <= 0:
        raise DownloadError(f"PNG 文件不完整或分辨率解析失败: {path.name}")
    return width, height


def _download_image(url: str, destination: Path, cfg: ImageConfig) -> tuple:
    if not url:
        raise DownloadError("图片 URL 为空")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise DownloadError(f"不支持的图片 URL: {url}")

    temp_path = destination.with_name(destination.name + ".part")
    last_error: Optional[DownloadError] = None

    for attempt in range(1, cfg.max_attempts + 1):
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(
                url,
                stream=True,
                timeout=(CONNECT_TIMEOUT, cfg.download_timeout),
            ) as response:
                if response.status_code >= 400:
                    _, _, _, detail = _response_error_parts(response)
                    raise DownloadError(
                        f"下载图片失败: {detail}",
                        retryable=response.status_code in RETRYABLE_HTTP_STATUS,
                    )
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            handle.write(chunk)

            width, height = _read_png_info(temp_path)
            file_size = temp_path.stat().st_size
            if file_size <= 0:
                raise DownloadError("下载到的图片内容为空（0 字节）")
            temp_path.replace(destination)
            return width, height, file_size

        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as exc:
            last_error = DownloadError(f"下载图片异常: {exc}", retryable=True)
            _remove_quietly(temp_path)
        except DownloadError as exc:
            last_error = exc
            _remove_quietly(temp_path)
            if not exc.retryable:
                raise
        except OSError as exc:
            _remove_quietly(temp_path)
            raise DownloadError(f"写入图片文件失败: {exc}") from exc

        if attempt < cfg.max_attempts:
            delay = _retry_delay(attempt, cfg)
            logger.warning(
                "下载图片失败，%.1fs 后重试（第 %d/%d 次）: %s",
                delay,
                attempt,
                cfg.max_retries,
                last_error,
            )
            time.sleep(delay)

    raise last_error or DownloadError("下载图片失败")


def generate_image(
    prompt: str,
    cfg: ImageConfig,
    output_path: Optional[Path] = None,
    output_root: Optional[Path] = None,
    scene_id: Optional[str] = None,
    filename: Optional[str] = None,
    scene_index: Optional[int] = None,
    ref_images: Optional[list] = None,
) -> ImageResult:
    """生成一张 PNG 图片，并自动下载保存到本地。ref_images 为参考图 data URI 列表。"""
    destination = _prepare_destination(
        output_path=output_path,
        output_root=output_root,
        scene_id=scene_id,
        filename=filename,
        scene_index=scene_index,
    )
    payload = build_payload(prompt, cfg, ref_images)
    requested_size = str(payload["parameters"]["size"])
    prompt_text = next((c["text"] for c in payload["input"]["messages"][0]["content"] if "text" in c), "")
    started = time.perf_counter()
    last_error: Optional[TaskError] = None

    for attempt in range(1, cfg.max_attempts + 1):
        try:
            logger.info(
                "→ 请求生成图片: model=%s size=%s chars=%d attempt=%d/%d dest=%s",
                cfg.model,
                requested_size,
                len(prompt_text),
                attempt,
                cfg.max_attempts,
                destination,
            )
            logger.debug("payload=%s", json.dumps(payload, ensure_ascii=False))

            task_id, create_request_id = _create_task(payload, cfg)
            logger.info("  任务已创建: task_id=%s request_id=%s", task_id, create_request_id)

            task_data = _poll_task(task_id, cfg)
            image_url, reported_size, query_request_id = _extract_image_result(task_data)
            width, height, file_size = _download_image(image_url, destination, cfg)

            duration_seconds = round(time.perf_counter() - started, 2)
            result = ImageResult(
                image_path=destination,
                requested_size=requested_size,
                width=width,
                height=height,
                actual_size=f"{width}*{height}",
                file_size=file_size,
                duration_seconds=duration_seconds,
                request_id=query_request_id or create_request_id,
                task_id=task_id,
                image_url=image_url,
                model=cfg.model,
                scene_id=scene_id,
                reported_size=reported_size,
                config_snapshot=cfg.safe_snapshot(),
            )
            logger.info(
                "✓ 图片完成: %s | %s | %s | %.2fs",
                result.image_path,
                result.actual_size,
                _human_size(result.file_size),
                result.duration_seconds,
            )
            return result

        except TaskError as exc:
            last_error = exc
            if not exc.retryable or attempt >= cfg.max_attempts:
                raise
            delay = _retry_delay(attempt, cfg)
            logger.warning(
                "生图任务失败，%.1fs 后重新提交（第 %d/%d 次）: %s",
                delay,
                attempt,
                cfg.max_retries,
                exc,
            )
            time.sleep(delay)

    raise last_error or ImageGenerationError("图片生成失败")


def _image_to_data_uri(path: Path) -> str:
    """把本地图片读成 data URI（wan2.7 多图参考用）。"""
    raw = Path(path).read_bytes()
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "bmp": "image/bmp"}.get(ext, "image/png")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def generate_project_images(
    project_dir: Path,
    cfg: Optional[ImageConfig] = None,
    env_path: Optional[Path] = None,
    stage: str = "all",
    resume: bool = True,
    **overrides: Any,
) -> dict:
    """短剧流水线：生成角色定妆照 + 逐 shot 关键帧（多图参考保证人物一致），回写 script.json。

    - stage=chars：每个角色文生图定妆照 → assets/chars/<key>.png，回写 character.ref_image；
    - stage=keyframes：每个 shot 以【角色定妆照(base64) + build_keyframe_prompt】多图参考生图
      → keyframes/<shot_id>.png，回写 status.keyframe/outputs.keyframe。
    """
    pdir = Path(project_dir).expanduser().resolve()
    script = project_io.load_script(pdir)
    if cfg is None:
        cfg = load_config(env_path, **overrides)
    try:
        cfg.size = normalize_size(prompt_kit.resolution_for(script), cfg.model)
        logger.info("使用画幅分辨率: %s", cfg.size)
    except ConfigError as exc:
        logger.warning("分辨率归一失败，沿用 %s: %s", cfg.size, exc)
    project_io.ensure_dirs(pdir)

    char_results: list = []
    key_results: list = []
    started = time.perf_counter()

    if stage in ("chars", "all"):
        for key, char in script.characters.items():
            out = project_io.char_portrait_path(pdir, key)
            if resume and out.exists() and out.stat().st_size > 0:
                logger.info("→ 角色 %s 定妆照已存在，跳过", key)
            else:
                _remove_quietly(out)
                prompt = prompt_kit.build_character_prompt(script, key)
                logger.info("→ 生成角色定妆照 %s", key)
                generate_image(prompt, cfg, output_path=out)
            char.ref_image = project_io.rel(pdir, out)
            char_results.append({"key": key, "path": str(out)})
        project_io.save_script(script, pdir / "script.json")

    if stage in ("keyframes", "all"):
        shots = list(script.iter_shots())
        for i, (scene, shot) in enumerate(shots, start=1):
            out = project_io.keyframe_path(pdir, shot.id)
            if resume and out.exists() and out.stat().st_size > 0:
                logger.info("→ 关键帧 %s（%d/%d）已存在，跳过", shot.id, i, len(shots))
                shot.status.keyframe = "done"
                shot.outputs.keyframe = project_io.rel(pdir, out)
                continue
            _remove_quietly(out)
            refs = []
            for ckey in shot.characters:
                c = script.characters.get(ckey)
                if c and c.ref_image:
                    rp = pdir / c.ref_image
                    if rp.exists():
                        refs.append(_image_to_data_uri(rp))
            prompt = prompt_kit.build_keyframe_prompt(script, scene, shot)
            logger.info("→ 关键帧 %s（%d/%d）参考图=%d", shot.id, i, len(shots), len(refs))
            try:
                generate_image(prompt, cfg, output_path=out, ref_images=refs or None)
                shot.status.keyframe = "done"
                shot.outputs.keyframe = project_io.rel(pdir, out)
                key_results.append({"shot_id": shot.id, "path": str(out), "refs": len(refs)})
            except ImageGenerationError as exc:
                shot.status.keyframe = "failed"
                logger.error("✗ 关键帧 %s 生成失败: %s", shot.id, exc)
        project_io.save_script(script, pdir / "script.json")

    return {"project_dir": pdir, "chars": char_results, "keyframes": key_results,
            "elapsed": round(time.perf_counter() - started, 2)}


def _image_result_to_dict(result: ImageResult) -> dict:
    return {
        "scene_id": result.scene_id,
        "image_path": str(result.image_path.resolve()),
        "requested_size": result.requested_size,
        "actual_size": result.actual_size,
        "width": result.width,
        "height": result.height,
        "file_size": result.file_size,
        "file_size_text": _human_size(result.file_size),
        "duration_seconds": result.duration_seconds,
        "request_id": result.request_id,
        "task_id": result.task_id,
        "image_url": result.image_url,
        "model": result.model,
    }


def _log_result_summary(results: Iterable[ImageResult]) -> None:
    logger.info("— 结果 —")
    for result in results:
        prefix = f"分镜 {result.scene_id} | " if result.scene_id else ""
        logger.info("  %s耗时      : %.2fs", prefix, result.duration_seconds)
        logger.info("    图片路径  : %s", result.image_path.resolve())
        logger.info("    实际分辨率: %s", result.actual_size)
        logger.info("    文件大小  : %s (%d bytes)", _human_size(result.file_size), result.file_size)
        logger.info("    request_id: %s", result.request_id)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image.generate",
        description="阿里云百炼 wan2.7-image-pro 图片生成服务",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="直接传入图片提示词")
    source.add_argument("--prompt-file", help="从文本文件读取图片提示词")
    source.add_argument("--project", help="短剧项目目录（projects/<id>）：读 script.json，生成 assets/chars 定妆照 + keyframes 关键帧")
    parser.add_argument("--consistency-seed", type=int,
                        help="对所有镜头使用固定 seed 以提升人物一致性（Shot 模式）")
    parser.add_argument("--no-resume", action="store_true",
                        help="不跳过已存在的图片，强制重新生成")
    parser.add_argument("--stage", choices=["chars", "keyframes", "all"], default="all",
                        help="--project 模式：chars=只生成定妆照, keyframes=只生成关键帧, all=两者（默认）")
    parser.add_argument(
        "--size",
        "--resolution",
        dest="size",
        help="输出尺寸：1K / 2K / 4K，或 WIDTHxHEIGHT、WIDTH*HEIGHT",
    )
    parser.add_argument("--model", help=f"模型名称（默认 {MODEL_NAME}）")
    parser.add_argument("--seed", type=int, help=f"随机种子 0-{MAX_SEED}")
    parser.add_argument(
        "--watermark",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否添加 AI 水印；同时提供 --no-watermark",
    )
    parser.add_argument(
        "--thinking",
        dest="thinking_mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否开启思考模式；同时提供 --no-thinking",
    )
    parser.add_argument("--output", help=f"输出根目录（默认 {DEFAULT_OUTPUT_ROOT}/<时间戳>）")
    parser.add_argument("--filename", help="单张图片输出文件名；非 .png 后缀会被替换")
    parser.add_argument("--env", help="指定 .env 文件路径")
    parser.add_argument("--request-timeout", type=float, help="单次 HTTP 请求读取超时（秒）")
    parser.add_argument("--task-timeout", type=float, help="单个生图任务轮询总超时（秒）")
    parser.add_argument("--poll-interval", type=float, help="任务轮询间隔（秒）")
    parser.add_argument("--download-timeout", type=float, help="图片下载超时（秒）")
    parser.add_argument("--max-retries", type=int, help="API、任务和下载的额外重试次数")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
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
        overrides = {
            "model": args.model,
            "size": args.size,
            "seed": args.seed,
            "watermark": args.watermark,
            "thinking_mode": args.thinking_mode,
            "request_timeout": args.request_timeout,
            "task_timeout": args.task_timeout,
            "poll_interval": args.poll_interval,
            "download_timeout": args.download_timeout,
            "max_retries": args.max_retries,
        }
        env_path = Path(args.env).expanduser() if args.env else None
        output_root = Path(args.output).expanduser() if args.output else None

        # ---- 短剧项目模式（projects/<id>）----
        if args.project:
            if args.consistency_seed is not None:
                overrides["seed"] = args.consistency_seed
            res = generate_project_images(
                Path(args.project).expanduser(), env_path=env_path, stage=args.stage,
                resume=not args.no_resume, **overrides)
            logger.info("— 结果 —")
            logger.info("  角色定妆照: %d | 关键帧: %d | 耗时 %.1fs",
                        len(res["chars"]), len(res["keyframes"]), res["elapsed"])
            print(json.dumps({
                "ok": True, "mode": "project",
                "project_dir": str(res["project_dir"].resolve()),
                "chars": len(res["chars"]), "keyframes": len(res["keyframes"]),
                "elapsed_seconds": res["elapsed"],
            }, ensure_ascii=False))
            return 0

        # ---- 单图模式（--prompt / --prompt-file）----
        if args.prompt:
            prompt = args.prompt
        else:
            prompt_file = Path(args.prompt_file).expanduser()
            if not prompt_file.exists():
                raise ImageGenerationError(f"提示词文件不存在: {prompt_file}")
            prompt = prompt_file.read_text(encoding="utf-8").strip()

        cfg = load_config(env_path, **overrides)
        result = generate_image(prompt, cfg, output_root=output_root, filename=args.filename)
        payload = {
            "ok": True,
            "mode": "prompt",
            "output_dir": str(result.image_path.parent.resolve()),
            "requested_size": result.requested_size,
            "model": result.model,
            "image_count": 1,
            "duration_seconds": result.duration_seconds,
            "images": [_image_result_to_dict(result)],
        }
        _log_result_summary([result])
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    except ConfigError as exc:
        logger.error("配置错误: %s", exc)
        print(json.dumps({"ok": False, "error_type": "ConfigError", "error": str(exc)}, ensure_ascii=False))
        return 3
    except ScriptError as exc:
        logger.error("脚本错误: %s", exc)
        print(json.dumps({"ok": False, "error_type": "ScriptError", "error": str(exc)}, ensure_ascii=False))
        return 4
    except ImageGenerationError as exc:
        logger.error("图片生成失败: %s", exc)
        print(json.dumps({"ok": False, "error_type": exc.__class__.__name__, "error": str(exc)}, ensure_ascii=False))
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
