#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阿里云百炼 wan3.0-video 图生视频服务模块（流水线 step 4）。

对 ``01-voice`` 归位后的 ``asset_type=video`` 镜头，用 wan3.0-video 的「首帧生视频」
（image-to-video）能力生成真实动态片段：以 ``02-image`` 产出的关键帧 ``image_<id>.png``
作为首帧（保证人物/场景一致性），``shot.video_prompt`` 描述动作与运镜。

要点：
- 采用官方 HTTP 异步接口：创建任务 -> 轮询 -> 下载 ``output.video_url``（24h 过期，立即下载）；
- API Key / 接入点从 ``.env`` 读取，端点构造与 ``02-image`` 一致
  （``DASHSCOPE_BASE_URL`` 主机 或 ``{WorkspaceId}.{region}.maas.aliyuncs.com`` + 原生路径）；
- 首帧用 ffmpeg 转为无 alpha 的 JPEG（scale/crop 到目标分辨率）再 base64，避免 PNG 透明通道与超大体积；
- ``audio=false``：生成片段不含音轨，声音统一由旁白 + BGM 在 step 7 混入；
- ``duration=clamp(ceil(shot.duration), 2, VIDEO_MAX_DUR)``（API 时长为整数、最小 2s），
  step 6 再精确裁剪到 shot.duration；
- 多线程并发 + 重试 + 断点续跑；单镜头失败不中断流水线，记为 ``failed``，由 step 6 回退为图片运镜；
- 输出 ``shot_<id>.mp4`` 与 ``videogen.json`` 清单。

CLI 示例::

    python 04-videogen/generate.py \
        --shots 03-shot/output/20261125_135634/shots.json \
        --images 02-image/output/20261126_090000

    # 只看 payload 不调用；仅生成部分镜头；小样省成本
    python 04-videogen/generate.py --shots .../shots.json --images ... \
        --dry-run --shot 7,8 --resolution 480P --concurrency 2

作为模块调用::

    from videogen.generate import load_config, generate_videos_from_shots
    cfg = load_config()
    result = generate_videos_from_shots(".../shots.json", images_dir=".../images", cfg=cfg)
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote, urlparse

try:
    import requests
except ImportError:  # pragma: no cover
    sys.stderr.write("缺少依赖 requests，请先执行: pip install -r 03-video/requirements.txt\n")
    raise

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import project as project_io  # noqa: E402
from common import prompts as prompt_kit  # noqa: E402


logger = logging.getLogger("videogen.generate")

# --------------------------------------------------------------------------- #
# 常量与默认值
# --------------------------------------------------------------------------- #
MODEL_NAME = "wan3.0-video"
SUPPORTED_MODELS = ("wan3.0-video", "wan3.0-video-prime")
DEFAULT_RESOLUTION = "1080P"
RESOLUTIONS = ("480P", "720P", "1080P")
DEFAULT_RATIO = "9:16"
RATIOS = ("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")

DEFAULT_BASE_ORIGIN = "https://dashscope.aliyuncs.com"
VIDEO_SYNTHESIS_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
TASK_PATH_PREFIX = "/api/v1/tasks/"

CONNECT_TIMEOUT = 10.0
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_PROMPT_CHARS = 20000
API_MIN_DURATION = 2
API_MAX_DURATION = 30
DEFAULT_MAX_DUR = 5                 # 单镜头生成时长上限（控成本），>= ceil(shot.duration) 即可
DEFAULT_TASK_TIMEOUT = 1200.0       # 视频生成通常 1-5 分钟，给足轮询上限
DEFAULT_POLL_INTERVAL = 15.0        # 官方建议 ~15s
DEFAULT_CONCURRENCY = 3
MAX_SEED = 2147483647
MAX_FIRST_FRAME_BYTES = 20 * 1024 * 1024
JPEG_QUALITY = 3                    # ffmpeg -q:v（越小质量越高）
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
MAX_RETRY_DELAY = 30.0

RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
NON_RETRYABLE_ERROR_CODES = {
    "AccessDenied", "Arrearage", "BadRequest", "DataInspectionFailed", "Forbidden",
    "InvalidApiKey", "InvalidParameter", "InvalidWorkspace", "ModelNotFound",
    "UnsupportedOperation",
}
NON_RETRYABLE_MESSAGE_KEYWORDS = (
    "access denied", "arrearage", "content policy", "data inspection", "forbidden",
    "invalid api key", "invalid parameter", "invalidapikey", "invalidparameter",
    "moderation", "nsfw", "unsupported", "mutually exclusive",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = _SCRIPT_DIR / "output"
_PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_IMAGE_ROOT = _PROJECT_ROOT / "02-image" / "output"

_REGION_HOSTS = {
    "cn-beijing": "{ws}.cn-beijing.maas.aliyuncs.com",
    "ap-southeast-1": "{ws}.ap-southeast-1.maas.aliyuncs.com",
    "ap-northeast-1": "{ws}.ap-northeast-1.maas.aliyuncs.com",
    "eu-central-1": "{ws}.eu-central-1.maas.aliyuncs.com",
    "us-east-1": "{ws}.us-east-1.maas.aliyuncs.com",
    "cn-hongkong": "{ws}.cn-hongkong.maas.aliyuncs.com",
}


# --------------------------------------------------------------------------- #
# 异常体系
# --------------------------------------------------------------------------- #
class VideoGenError(Exception):
    """视频生成服务基础异常。"""


class ConfigError(VideoGenError):
    """配置缺失或非法。"""


class ScriptError(VideoGenError):
    """shots.json 解析失败。"""


class FFmpegNotFound(VideoGenError):
    """未找到可用的 ffmpeg 二进制。"""


class APIError(VideoGenError):
    """阿里云百炼 HTTP API 异常。"""

    def __init__(self, message: str, *, status_code: Optional[int] = None,
                 code: Optional[str] = None, request_id: Optional[str] = None,
                 retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.retryable = retryable


class TaskError(VideoGenError):
    """视频任务进入失败或未知状态。"""

    def __init__(self, message: str, *, task_id: str = "", task_status: str = "",
                 code: Optional[str] = None, request_id: Optional[str] = None,
                 retryable: bool = False) -> None:
        super().__init__(message)
        self.task_id = task_id
        self.task_status = task_status
        self.code = code
        self.request_id = request_id
        self.retryable = retryable


class TaskTimeoutError(TaskError):
    """视频任务轮询超时。"""

    def __init__(self, message: str, *, task_id: str = "", task_status: str = "",
                 request_id: Optional[str] = None) -> None:
        super().__init__(message, task_id=task_id, task_status=task_status,
                         request_id=request_id, retryable=False)


class DownloadError(VideoGenError):
    """视频下载或写盘异常。"""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class VideoConfig:
    """wan3.0-video 配置。敏感字段不会出现在快照中。"""

    api_key: str
    base_url: Optional[str] = None
    workspace_id: str = ""
    region: str = "cn-beijing"
    model: str = MODEL_NAME
    resolution: str = DEFAULT_RESOLUTION
    ratio: str = DEFAULT_RATIO
    max_duration: int = DEFAULT_MAX_DUR
    audio: bool = False
    prompt_extend: bool = True
    watermark: bool = False
    seed: Optional[int] = None
    concurrency: int = DEFAULT_CONCURRENCY
    request_timeout: float = 120.0
    task_timeout: float = DEFAULT_TASK_TIMEOUT
    poll_interval: float = DEFAULT_POLL_INTERVAL
    download_timeout: float = 300.0
    max_retries: int = 2
    retry_backoff: float = 3.0

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
        return f"{self.api_origin}{VIDEO_SYNTHESIS_PATH}"

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
class ShotTask:
    """一个待生成视频的镜头。"""

    shot_id: Any
    scene_id: Any
    duration: float
    video_prompt: str
    keyframe_path: Optional[Path] = None


@dataclass
class ClipResult:
    """单个镜头的生成结果。"""

    shot_id: Any
    status: str                       # ok / skipped / failed
    video_path: Optional[Path]
    requested_duration: int
    task_id: str = ""
    video_url: str = ""
    elapsed_seconds: float = 0.0
    file_size: int = 0
    error: str = ""


@dataclass
class BatchResult:
    """整批生成结果。"""

    output_dir: Path
    clips: list
    manifest_path: Optional[Path]
    total_elapsed_seconds: float
    ok_count: int
    failed_count: int
    skipped_count: int
    model: str
    config_snapshot: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
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
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


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
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"环境变量 {name} 不是合法布尔值: {raw!r}")


def _env_number(name: str, default: Any, cast: type) -> Any:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"环境变量 {name} 不是合法数字: {raw!r}") from exc


def _resolve_env_path(explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None
    for c in (_SCRIPT_DIR / ".env", _SCRIPT_DIR.parent / ".env", Path.cwd() / ".env"):
        if c.exists():
            return c
    return None


def resolve_ffmpeg(explicit: Optional[str] = None) -> str:
    """按优先级解析 ffmpeg 可执行文件路径（首帧转码用）。"""
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
        "未找到 ffmpeg（首帧转 JPEG 需要）。请 pip install imageio-ffmpeg 或 brew install ffmpeg，"
        "或用 --ffmpeg 指定。"
    )


def _parse_resolution(value: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d{2,5})\s*[xX*×:]\s*(\d{2,5})\s*", value or "")
    if not m:
        raise VideoGenError(f"分辨率格式非法: {value!r}，应形如 1080x1920")
    return int(m.group(1)), int(m.group(2))


def latest_image_dir(image_root: Optional[Path] = None) -> Optional[Path]:
    """自动挑选 02-image/output 下最新时间戳目录。"""
    root = Path(image_root) if image_root else DEFAULT_IMAGE_ROOT
    if not root.is_dir():
        return None
    dirs = [d for d in root.iterdir() if d.is_dir() and re.fullmatch(r"\d{8}_\d{6}", d.name)]
    return max(dirs, key=lambda d: d.name) if dirs else None


def _find_keyframe(images_dir: Path, shot_id: Any) -> Optional[Path]:
    """在关键帧目录中按 shot id 匹配图片（image_001 / image_1 / image_01）。"""
    key = str(shot_id).strip()
    stems = []
    if key.isdigit():
        n = int(key)
        stems = [f"image_{n:03d}", f"image_{n}", f"image_{n:02d}"]
    else:
        stems = [f"image_{key}"]
    for stem in stems:
        for ext in IMAGE_EXTS:
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
    return None


# --------------------------------------------------------------------------- #
# 配置加载
# --------------------------------------------------------------------------- #
def _validate_config(cfg: VideoConfig) -> None:
    if not cfg.api_key:
        raise ConfigError("缺少 DASHSCOPE_API_KEY")
    if cfg.model not in SUPPORTED_MODELS:
        raise ConfigError(f"不支持的模型: {cfg.model!r}，可选: {SUPPORTED_MODELS}")
    if cfg.resolution not in RESOLUTIONS:
        raise ConfigError(f"resolution 需为 {RESOLUTIONS} 之一，当前 {cfg.resolution!r}")
    if cfg.ratio not in RATIOS:
        raise ConfigError(f"ratio 需为 {RATIOS} 之一，当前 {cfg.ratio!r}")
    if not API_MIN_DURATION <= cfg.max_duration <= API_MAX_DURATION:
        raise ConfigError(f"max_duration 需在 [{API_MIN_DURATION},{API_MAX_DURATION}]，当前 {cfg.max_duration}")
    if cfg.concurrency < 1:
        raise ConfigError(f"concurrency 需 >= 1，当前 {cfg.concurrency}")
    if cfg.task_timeout <= 0 or cfg.poll_interval <= 0:
        raise ConfigError("task_timeout / poll_interval 必须大于 0")
    if cfg.max_retries < 0 or cfg.retry_backoff < 0:
        raise ConfigError("max_retries / retry_backoff 不能为负")
    if cfg.seed is not None and not (cfg.seed == -1 or 0 <= cfg.seed <= MAX_SEED):
        raise ConfigError(f"seed 需为 -1 或 [0,{MAX_SEED}]，当前 {cfg.seed}")
    cfg.api_origin  # 触发端点校验


def load_config(env_path: Optional[Path] = None, **overrides: Any) -> VideoConfig:
    """从 ``.env`` 加载配置，并用 ``overrides`` 中非 None 的值覆盖。

    环境变量：
        DASHSCOPE_API_KEY        必填
        DASHSCOPE_BASE_URL       可选，显式接入点主机（会拼接原生 video-synthesis 路径）
        DASHSCOPE_WORKSPACE_ID   可选，业务空间 ID（拼接专属域名）
        DASHSCOPE_REGION         可选，默认 cn-beijing
        VIDEO_MODEL / VIDEO_RESOLUTION / VIDEO_RATIO / VIDEO_MAX_DUR
        VIDEO_AUDIO / VIDEO_PROMPT_EXTEND / VIDEO_WATERMARK / VIDEO_SEED
        VIDEO_CONCURRENCY / VIDEO_REQUEST_TIMEOUT / VIDEO_TASK_TIMEOUT
        VIDEO_POLL_INTERVAL / VIDEO_DOWNLOAD_TIMEOUT / VIDEO_MAX_RETRIES / VIDEO_RETRY_BACKOFF
    """
    resolved = _resolve_env_path(env_path)
    if load_dotenv is not None:
        load_dotenv(resolved, override=False) if resolved else load_dotenv(override=False)
    elif resolved:
        logger.warning("未安装 python-dotenv，将仅使用进程环境变量（忽略 %s）", resolved)

    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        raise ConfigError("缺少 DASHSCOPE_API_KEY，请在 .env 中配置（参考 03-video/.env.example）")

    kwargs: dict = {
        "api_key": api_key,
        "base_url": _env_optional("DASHSCOPE_BASE_URL"),
        "workspace_id": (os.getenv("DASHSCOPE_WORKSPACE_ID") or "").strip(),
        "region": (os.getenv("DASHSCOPE_REGION") or "cn-beijing").strip() or "cn-beijing",
        "model": (os.getenv("VIDEO_MODEL") or MODEL_NAME).strip() or MODEL_NAME,
        "resolution": (os.getenv("VIDEO_RESOLUTION") or DEFAULT_RESOLUTION).strip() or DEFAULT_RESOLUTION,
        "ratio": (os.getenv("VIDEO_RATIO") or DEFAULT_RATIO).strip() or DEFAULT_RATIO,
        "max_duration": _env_number("VIDEO_MAX_DUR", DEFAULT_MAX_DUR, int),
        "audio": _env_bool("VIDEO_AUDIO", False),
        "prompt_extend": _env_bool("VIDEO_PROMPT_EXTEND", True),
        "watermark": _env_bool("VIDEO_WATERMARK", False),
        "concurrency": _env_number("VIDEO_CONCURRENCY", DEFAULT_CONCURRENCY, int),
        "request_timeout": _env_number("VIDEO_REQUEST_TIMEOUT", 120.0, float),
        "task_timeout": _env_number("VIDEO_TASK_TIMEOUT", DEFAULT_TASK_TIMEOUT, float),
        "poll_interval": _env_number("VIDEO_POLL_INTERVAL", DEFAULT_POLL_INTERVAL, float),
        "download_timeout": _env_number("VIDEO_DOWNLOAD_TIMEOUT", 300.0, float),
        "max_retries": _env_number("VIDEO_MAX_RETRIES", 2, int),
        "retry_backoff": _env_number("VIDEO_RETRY_BACKOFF", 3.0, float),
    }
    seed_raw = os.getenv("VIDEO_SEED")
    if seed_raw and seed_raw.strip():
        kwargs["seed"] = int(seed_raw.strip())

    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    cfg = VideoConfig(**kwargs)
    _validate_config(cfg)
    return cfg


# --------------------------------------------------------------------------- #
# 读取 shots.json
# --------------------------------------------------------------------------- #
def _normalize_ids(values: Iterable[Any]) -> set:
    out = set()
    for v in values:
        for part in str(v).replace("，", ",").split(","):
            part = part.strip()
            if part:
                out.add(str(int(part)) if part.isdigit() else part)
    return out


def load_video_tasks(project_dir: Path, shot_ids: Optional[Iterable[Any]] = None,
                     skip_motion_less: bool = True) -> tuple:
    """读 projects/<id>/script.json，返回 (meta, [ShotTask])。

    模板无 asset_type：默认所有有 motion 的 shot 都生成 AI 视频（skip_motion_less=True 时
    跳过 motion 为空的静态镜头）；时长优先用 runtime_sec（真实配音），否则名义 duration_sec。
    """
    pdir = Path(project_dir).expanduser().resolve()
    script = project_io.load_script(pdir)
    meta = {"path": pdir / "script.json",
            "resolution": prompt_kit.resolution_for(script),
            "aspect_ratio": script.project.aspect_ratio}
    tasks: list[ShotTask] = []
    for scene, shot in script.iter_shots():
        if skip_motion_less and not (shot.motion or "").strip():
            continue
        dur = shot.runtime_sec if shot.runtime_sec is not None else shot.duration_sec
        tasks.append(ShotTask(
            shot_id=shot.id,
            scene_id=scene.id,
            duration=float(dur or 0),
            video_prompt=prompt_kit.build_video_prompt(script, scene, shot),
            keyframe_path=project_io.keyframe_path(pdir, shot.id),
        ))
    if shot_ids:
        wanted = _normalize_ids(shot_ids)
        tasks = [t for t in tasks if str(t.shot_id) in wanted]
    return meta, tasks


# --------------------------------------------------------------------------- #
# 首帧编码（PNG -> 无 alpha JPEG -> base64 data URI）
# --------------------------------------------------------------------------- #
def encode_first_frame(image_path: Path, ffmpeg_bin: str, width: int, height: int,
                       work_dir: Path) -> tuple[str, int]:
    """把关键帧转为目标分辨率的无 alpha JPEG 并 base64，返回 (data_uri, jpeg_bytes_size)。"""
    if not image_path.exists():
        raise DownloadError(f"关键帧不存在: {image_path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    jpg = work_dir / f"{image_path.stem}_ff.jpg"
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    cmd = [ffmpeg_bin, "-y", "-i", str(image_path), "-vf", vf,
           "-frames:v", "1", "-q:v", str(JPEG_QUALITY), str(jpg)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VideoGenError(f"首帧转码调用 ffmpeg 失败: {exc}") from exc
    if proc.returncode != 0 or not jpg.exists():
        tail = (proc.stderr or "").strip().splitlines()[-4:]
        raise VideoGenError(f"首帧转码失败 (rc={proc.returncode}):\n  " + "\n  ".join(tail))
    raw = jpg.read_bytes()
    size = len(raw)
    if size <= 0:
        raise DownloadError("首帧 JPEG 为空")
    if size > MAX_FIRST_FRAME_BYTES:
        raise DownloadError(f"首帧 JPEG 超过 {MAX_FIRST_FRAME_BYTES} 字节: {size}")
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", size


def compute_request_duration(shot_duration: float, cfg: VideoConfig) -> int:
    """API duration 为整数、最小 2s：取 clamp(ceil(shot_duration), 2, max_duration)。"""
    d = int(math.ceil(max(shot_duration, 0.0)))
    return max(API_MIN_DURATION, min(cfg.max_duration, max(d, API_MIN_DURATION)))


def build_payload(prompt: str, first_frame_uri: Optional[str], req_duration: int,
                  cfg: VideoConfig) -> dict:
    """构造 wan3.0-video 首帧生视频请求体。"""
    text = (prompt or "").strip()
    if not text:
        raise VideoGenError("video_prompt 为空")
    if len(text) > MAX_PROMPT_CHARS:
        text = text[:MAX_PROMPT_CHARS]
    inp: dict = {"prompt": text}
    if first_frame_uri:
        inp["media"] = [{"type": "first_frame", "url": first_frame_uri}]
    params: dict = {
        "resolution": cfg.resolution,
        "ratio": cfg.ratio,
        "duration": req_duration,
        "audio": bool(cfg.audio),
        "prompt_extend": bool(cfg.prompt_extend),
        "watermark": bool(cfg.watermark),
    }
    if cfg.seed is not None:
        params["seed"] = cfg.seed
    return {"model": cfg.model, "input": inp, "parameters": params}


# --------------------------------------------------------------------------- #
# HTTP：创建任务 / 轮询 / 下载（含重试）
# --------------------------------------------------------------------------- #
def _is_retryable(code: Optional[str], message: Optional[str]) -> bool:
    if code and code in NON_RETRYABLE_ERROR_CODES:
        return False
    text = f"{code or ''} {message or ''}".lower()
    return not any(k in text for k in NON_RETRYABLE_MESSAGE_KEYWORDS)


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
    detail = (f"HTTP {response.status_code} | code={code} | message={message} | request_id={request_id}"
              if (code or message) else f"HTTP {response.status_code} | {text}")
    return code, message, request_id, detail


def _retry_delay(attempt: int, cfg: VideoConfig, retry_after: Optional[float] = None) -> float:
    if retry_after is not None:
        return retry_after
    return min(cfg.retry_backoff * (2 ** (attempt - 1)), MAX_RETRY_DELAY)


def _send_json_request(method: str, url: str, cfg: VideoConfig, operation: str,
                       headers: Optional[dict] = None, payload: Optional[dict] = None,
                       timeout: Optional[tuple] = None, raise_on_body_code: bool = True) -> dict:
    request_timeout = timeout or cfg.http_timeout
    last_error: Optional[APIError] = None
    for attempt in range(1, cfg.max_attempts + 1):
        retry_after: Optional[float] = None
        try:
            response = requests.request(method, url, headers=headers, json=payload,
                                        timeout=request_timeout)
        except requests.exceptions.Timeout as exc:
            last_error = APIError(f"{operation}超时（>{request_timeout[1]}s）: {exc}", retryable=True)
        except requests.exceptions.RequestException as exc:
            last_error = APIError(f"{operation}网络异常: {exc}", retryable=True)
        else:
            if response.status_code in RETRYABLE_HTTP_STATUS:
                code, _, request_id, detail = _response_error_parts(response)
                raw_ra = response.headers.get("Retry-After") if hasattr(response.headers, "get") else None
                try:
                    retry_after = min(max(float(raw_ra), 0.0), MAX_RETRY_DELAY) if raw_ra else None
                except (TypeError, ValueError):
                    retry_after = None
                last_error = APIError(f"{operation}服务暂时不可用: {detail}",
                                      status_code=response.status_code, code=code,
                                      request_id=request_id, retryable=True)
            elif response.status_code >= 400:
                code, _, request_id, detail = _response_error_parts(response)
                raise APIError(f"{operation}接口返回错误: {detail}", status_code=response.status_code,
                               code=code, request_id=request_id, retryable=False)
            else:
                try:
                    data = response.json()
                except ValueError as exc:
                    raise APIError(f"{operation}响应不是合法 JSON: {(response.text or '')[:300]}") from exc
                if not isinstance(data, dict):
                    raise APIError(f"{operation}响应格式非法: {str(data)[:300]}")
                code = data.get("code") or data.get("Code")
                if raise_on_body_code and code:
                    message = data.get("message") or data.get("Message")
                    request_id = data.get("request_id") or data.get("RequestId")
                    retryable = _is_retryable(str(code), str(message or ""))
                    err = APIError(f"{operation}返回业务错误: code={code} | message={message}",
                                   status_code=response.status_code, code=str(code),
                                   request_id=str(request_id or ""), retryable=retryable)
                    if retryable:
                        last_error = err
                    else:
                        raise err
                return data
        if attempt < cfg.max_attempts:
            delay = _retry_delay(attempt, cfg, retry_after)
            logger.warning("%s失败，%.1fs 后重试（%d/%d）: %s", operation, delay, attempt, cfg.max_retries, last_error)
            time.sleep(delay)
    raise last_error or APIError(f"{operation}失败: 未知错误")


def _create_task(payload: dict, cfg: VideoConfig) -> tuple[str, str]:
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json",
               "X-DashScope-Async": "enable"}
    data = _send_json_request("POST", cfg.create_endpoint, cfg, "创建视频任务",
                              headers=headers, payload=payload)
    output = data.get("output") if isinstance(data.get("output"), dict) else {}
    task_id = output.get("task_id") or data.get("task_id")
    request_id = data.get("request_id") or ""
    if not task_id:
        raise APIError(f"创建任务响应缺少 output.task_id: {json.dumps(data, ensure_ascii=False)[:500]}")
    return str(task_id), str(request_id)


def _poll_task(task_id: str, cfg: VideoConfig) -> dict:
    url = cfg.task_endpoint(task_id)
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    deadline = time.monotonic() + cfg.task_timeout
    while True:
        data = _send_json_request("GET", url, cfg, "查询视频任务", headers=headers,
                                  raise_on_body_code=False)
        output = data.get("output") if isinstance(data.get("output"), dict) else {}
        status = str(output.get("task_status") or data.get("task_status") or "UNKNOWN").upper()
        request_id = str(data.get("request_id") or "")
        if status in {"PENDING", "RUNNING"}:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TaskTimeoutError(f"视频任务轮询超时（>{cfg.task_timeout}s），task_id={task_id}",
                                       task_id=task_id, task_status=status, request_id=request_id)
            time.sleep(min(cfg.poll_interval, remaining))
            continue
        if status == "SUCCEEDED":
            return data
        code = data.get("code") or output.get("code")
        message = data.get("message") or output.get("message") or output.get("status_message")
        retryable = status != "CANCELED" and _is_retryable(str(code or ""), str(message or ""))
        raise TaskError(f"视频任务失败: task_id={task_id} | status={status} | code={code} | message={message}",
                        task_id=task_id, task_status=status, code=str(code or "") or None,
                        request_id=request_id, retryable=retryable)


def _extract_video_url(data: dict) -> str:
    output = data.get("output") if isinstance(data.get("output"), dict) else {}
    url = output.get("video_url")
    if not url:
        for item in output.get("results") or []:
            if isinstance(item, dict) and (item.get("video_url") or item.get("url")):
                url = item.get("video_url") or item.get("url")
                break
    if not url:
        raise APIError(f"任务成功但未返回 video_url: {json.dumps(data, ensure_ascii=False)[:500]}")
    return str(url)


def _download_video(url: str, destination: Path, cfg: VideoConfig) -> int:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise DownloadError(f"不支持的视频 URL: {url}")
    temp_path = destination.with_name(destination.name + ".part")
    last_error: Optional[DownloadError] = None
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(url, stream=True, timeout=(CONNECT_TIMEOUT, cfg.download_timeout)) as resp:
                if resp.status_code >= 400:
                    _, _, _, detail = _response_error_parts(resp)
                    raise DownloadError(f"下载视频失败: {detail}",
                                        retryable=resp.status_code in RETRYABLE_HTTP_STATUS)
                with temp_path.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
            size = temp_path.stat().st_size
            if size <= 0:
                raise DownloadError("下载到的视频为空（0 字节）")
            head = temp_path.open("rb").read(16)
            if b"ftyp" not in head and not head.startswith(b"\x00\x00\x00"):
                logger.warning("下载内容可能不是标准 MP4（头部: %r），仍保存", head[:8])
            temp_path.replace(destination)
            return size
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as exc:
            last_error = DownloadError(f"下载视频异常: {exc}", retryable=True)
            _remove_quietly(temp_path)
        except DownloadError as exc:
            last_error = exc
            _remove_quietly(temp_path)
            if not exc.retryable:
                raise
        except OSError as exc:
            _remove_quietly(temp_path)
            raise DownloadError(f"写入视频文件失败: {exc}") from exc
        if attempt < cfg.max_attempts:
            delay = _retry_delay(attempt, cfg)
            logger.warning("下载视频失败，%.1fs 后重试（%d/%d）: %s", delay, attempt, cfg.max_retries, last_error)
            time.sleep(delay)
    raise last_error or DownloadError("下载视频失败")


# --------------------------------------------------------------------------- #
# 单镜头生成
# --------------------------------------------------------------------------- #
def _stem(shot_id: Any) -> str:
    """AI 片段文件名主干：ai_001 / ai_002 …"""
    key = str(shot_id).strip()
    return f"ai_{int(key):03d}" if key.isdigit() else f"ai_{re.sub(r'[^A-Za-z0-9_.-]+', '-', key).strip('-._')}"


def generate_clip(task: ShotTask, cfg: VideoConfig, *, out_dir: Path,
                  ffmpeg_bin: str, width: int, height: int, resume: bool = True,
                  dry_run: bool = False) -> ClipResult:
    """生成单个镜头视频。失败抛 VideoGenError，由上层记录为 failed。"""
    stem = _stem(task.shot_id)
    out_path = out_dir / f"{stem}.mp4"
    req_duration = compute_request_duration(task.duration, cfg)

    if resume and out_path.exists() and out_path.stat().st_size > 0:
        logger.info("→ 镜头 %s 已存在，跳过：%s", task.shot_id, out_path.name)
        return ClipResult(shot_id=task.shot_id, status="skipped", video_path=out_path,
                          requested_duration=req_duration, file_size=out_path.stat().st_size)

    keyframe = task.keyframe_path
    if keyframe is None or not Path(keyframe).exists():
        raise VideoGenError(f"镜头 {task.shot_id} 找不到关键帧 {keyframe}（请先跑 02-image --project）")

    if dry_run:
        logger.info("[dry-run] 镜头 %s first_frame=%s duration=%ds prompt_chars=%d → %s",
                    task.shot_id, keyframe.name, req_duration, len(task.video_prompt), out_path.name)
        return ClipResult(shot_id=task.shot_id, status="dry_run", video_path=out_path,
                          requested_duration=req_duration)

    started = time.perf_counter()
    ff_dir = out_dir / "_firstframes"
    first_frame_uri, jpeg_size = encode_first_frame(keyframe, ffmpeg_bin, width, height, ff_dir)
    logger.info("→ 镜头 %s 生成视频: model=%s res=%s ratio=%s duration=%ds first_frame=%s(%s)",
                task.shot_id, cfg.model, cfg.resolution, cfg.ratio, req_duration,
                keyframe.name, _human_size(jpeg_size))

    payload = build_payload(task.video_prompt, first_frame_uri, req_duration, cfg)
    last_error: Optional[Exception] = None
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            task_id, create_rid = _create_task(payload, cfg)
            logger.info("  镜头 %s 任务已创建: task_id=%s", task.shot_id, task_id)
            data = _poll_task(task_id, cfg)
            video_url = _extract_video_url(data)
            size = _download_video(video_url, out_path, cfg)
            elapsed = round(time.perf_counter() - started, 2)
            logger.info("✓ 镜头 %s: %s | %s | %.1fs", task.shot_id, out_path.name,
                        _human_size(size), elapsed)
            return ClipResult(shot_id=task.shot_id, status="ok", video_path=out_path,
                              requested_duration=req_duration, task_id=task_id,
                              video_url=video_url, elapsed_seconds=elapsed, file_size=size)
        except (TaskError, APIError) as exc:
            last_error = exc
            if not getattr(exc, "retryable", False) or attempt >= cfg.max_attempts:
                break
            delay = _retry_delay(attempt, cfg)
            logger.warning("镜头 %s 生成失败，%.1fs 后重试（%d/%d）: %s",
                           task.shot_id, delay, attempt, cfg.max_retries, exc)
            time.sleep(delay)
    raise last_error or VideoGenError(f"镜头 {task.shot_id} 生成失败")


# --------------------------------------------------------------------------- #
# 批量生成
# --------------------------------------------------------------------------- #
def generate_ai_clips(
    project_dir: Path,
    cfg: Optional[VideoConfig] = None,
    env_path: Optional[Path] = None,
    shot_ids: Optional[Iterable[Any]] = None,
    resume: bool = True,
    dry_run: bool = False,
    ffmpeg_bin: Optional[str] = None,
    skip_motion_less: bool = True,
    **overrides: Any,
) -> BatchResult:
    """并发为有 motion 的镜头生成 AI 片段到 projects/<id>/video/ai_<shot>.mp4，写 videogen.json。"""
    pdir = Path(project_dir).expanduser().resolve()
    meta, tasks = load_video_tasks(pdir, shot_ids, skip_motion_less)
    if not tasks:
        raise ScriptError("没有需要生成 AI 视频的镜头（motion 均为空？或用 --shot 指定）")
    if cfg is None:
        cfg = load_config(env_path, **overrides)
    ff = ffmpeg_bin or resolve_ffmpeg(None)
    try:
        width, height = _parse_resolution(meta.get("resolution") or "1080x1920")
    except VideoGenError:
        width, height = 1080, 1920
    out_dir = project_io.video_dir(pdir)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    clips: list[ClipResult] = []
    logger.info("待生成 AI 视频镜头: %d | 并发: %d | ffmpeg: %s", len(tasks), cfg.concurrency, ff)

    def _run(t: ShotTask) -> ClipResult:
        try:
            return generate_clip(t, cfg, out_dir=out_dir, ffmpeg_bin=ff, width=width,
                                 height=height, resume=resume, dry_run=dry_run)
        except VideoGenError as exc:
            logger.error("✗ 镜头 %s 生成失败（装配阶段将回退为关键帧运镜）: %s", t.shot_id, exc)
            return ClipResult(shot_id=t.shot_id, status="failed", video_path=None,
                              requested_duration=compute_request_duration(t.duration, cfg), error=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("镜头 %s 未预期错误: %s", t.shot_id, exc)
            return ClipResult(shot_id=t.shot_id, status="failed", video_path=None,
                              requested_duration=compute_request_duration(t.duration, cfg),
                              error=f"{exc.__class__.__name__}: {exc}")

    if cfg.concurrency <= 1 or len(tasks) == 1:
        for t in tasks:
            clips.append(_run(t))
    else:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            futures = {pool.submit(_run, t): t for t in tasks}
            for fut in as_completed(futures):
                clips.append(fut.result())
    clips.sort(key=lambda c: str(c.shot_id))

    ok = sum(1 for c in clips if c.status == "ok")
    failed = sum(1 for c in clips if c.status == "failed")
    skipped = sum(1 for c in clips if c.status == "skipped")

    manifest_path: Optional[Path] = None
    if not dry_run:
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "project_dir": str(pdir), "output_dir": str(out_dir.resolve()),
            "model": cfg.model, "resolution": cfg.resolution, "ratio": cfg.ratio,
            "audio": cfg.audio, "prompt_extend": cfg.prompt_extend,
            "total_elapsed_seconds": round(time.perf_counter() - started, 2),
            "ok": ok, "failed": failed, "skipped": skipped,
            "clips": [{"shot_id": c.shot_id, "status": c.status,
                       "video_file": c.video_path.name if c.video_path else None,
                       "requested_duration": c.requested_duration, "task_id": c.task_id,
                       "video_url": c.video_url, "elapsed_seconds": c.elapsed_seconds,
                       "file_size": c.file_size, "error": c.error} for c in clips],
        }
        manifest_path = out_dir / "videogen.json"
        try:
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("写入 videogen.json 失败: %s", exc)

    return BatchResult(
        output_dir=out_dir, clips=clips, manifest_path=manifest_path,
        total_elapsed_seconds=round(time.perf_counter() - started, 2),
        ok_count=ok, failed_count=failed, skipped_count=skipped,
        model=cfg.model, config_snapshot=cfg.safe_snapshot(),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="videogen.aigen",
        description="wan3.0-video 图生视频（为有 motion 的镜头生成 AI 片段 ai_<shot>.mp4）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project", required=True, help="短剧项目目录（projects/<id>，含 script.json）")
    parser.add_argument("--shot", action="append", help="仅生成指定镜头，可重复或逗号分隔")
    parser.add_argument("--model", help=f"模型名称（默认 {MODEL_NAME}）")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS), help="分辨率档位")
    parser.add_argument("--ratio", choices=list(RATIOS), help="宽高比")
    parser.add_argument("--max-duration", type=int, help=f"单镜头生成时长上限秒（[{API_MIN_DURATION},{API_MAX_DURATION}]）")
    parser.add_argument("--audio", action=argparse.BooleanOptionalAction, default=None,
                        help="生成片段是否含音轨（默认否）")
    parser.add_argument("--prompt-extend", dest="prompt_extend", action=argparse.BooleanOptionalAction,
                        default=None, help="是否开启 prompt 智能改写")
    parser.add_argument("--watermark", action=argparse.BooleanOptionalAction, default=None, help="是否加水印")
    parser.add_argument("--seed", type=int, help=f"随机种子 -1 或 [0,{MAX_SEED}]")
    parser.add_argument("--concurrency", type=int, help="并发任务数")
    parser.add_argument("--task-timeout", type=float, help="单任务轮询总超时（秒）")
    parser.add_argument("--poll-interval", type=float, help="轮询间隔（秒）")
    parser.add_argument("--max-retries", type=int, help="额外重试次数")
    parser.add_argument("--include-static", action="store_true", help="motion 为静的镜头也生成 AI 视频")
    parser.add_argument("--no-resume", action="store_true", help="不跳过已存在的 ai_<shot>.mp4，强制重生成")
    parser.add_argument("--ffmpeg", help="显式指定 ffmpeg 可执行文件路径")
    parser.add_argument("--env", help="指定 .env 文件路径")
    parser.add_argument("--strict", action="store_true", help="有镜头失败时以非零码退出")
    parser.add_argument("--dry-run", action="store_true", help="只构造并打印 payload，不调用 API")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S", stream=sys.stderr,
    )
    try:
        overrides = {
            "model": args.model, "resolution": args.resolution, "ratio": args.ratio,
            "max_duration": args.max_duration, "audio": args.audio,
            "prompt_extend": args.prompt_extend, "watermark": args.watermark,
            "seed": args.seed, "concurrency": args.concurrency,
            "task_timeout": args.task_timeout, "poll_interval": args.poll_interval,
            "max_retries": args.max_retries,
        }
        result = generate_ai_clips(
            Path(args.project).expanduser(), env_path=Path(args.env).expanduser() if args.env else None,
            shot_ids=args.shot, resume=not args.no_resume, dry_run=args.dry_run,
            ffmpeg_bin=resolve_ffmpeg(args.ffmpeg) if args.ffmpeg else None,
            skip_motion_less=not args.include_static, **overrides,
        )
        logger.info("— 结果 —")
        logger.info("  模型      : %s | %s %s", result.model,
                    result.config_snapshot.get("resolution"), result.config_snapshot.get("ratio"))
        logger.info("  成功/跳过/失败: %d / %d / %d", result.ok_count, result.skipped_count, result.failed_count)
        logger.info("  总耗时    : %.1fs", result.total_elapsed_seconds)
        if not args.dry_run:
            logger.info("  输出目录  : %s", result.output_dir.resolve())
        print(json.dumps({
            "ok": True, "dry_run": args.dry_run,
            "output_dir": str(result.output_dir.resolve()),
            "manifest": str(result.manifest_path.resolve()) if result.manifest_path else None,
            "model": result.model, "ok_count": result.ok_count,
            "skipped": result.skipped_count, "failed": result.failed_count,
            "clips": [{"shot_id": c.shot_id, "status": c.status,
                       "video_path": str(c.video_path.resolve()) if c.video_path else None,
                       "requested_duration": c.requested_duration, "error": c.error}
                      for c in result.clips],
        }, ensure_ascii=False))
        if args.strict and result.failed_count > 0:
            return 1
        return 0
    except ConfigError as exc:
        logger.error("配置错误: %s", exc)
        print(json.dumps({"ok": False, "error_type": "ConfigError", "error": str(exc)}, ensure_ascii=False))
        return 3
    except FFmpegNotFound as exc:
        logger.error("%s", exc)
        print(json.dumps({"ok": False, "error_type": "FFmpegNotFound", "error": str(exc)}, ensure_ascii=False))
        return 5
    except ScriptError as exc:
        logger.error("脚本错误: %s", exc)
        print(json.dumps({"ok": False, "error_type": "ScriptError", "error": str(exc)}, ensure_ascii=False))
        return 4
    except VideoGenError as exc:
        logger.error("视频生成失败: %s", exc)
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
