#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阿里云百炼 qwen-audio-3.0-tts-plus 非实时语音合成服务模块。

功能：
- 通过阿里云百炼 HTTP API 调用 ``qwen-audio-3.0-tts-plus`` 合成语音；
- API Key、接入点（Base URL / Workspace ID）从 ``.env`` 读取，禁止硬编码；
  同时兼容标准百炼网关与 Token Plan 网关（token-plan.cn-beijing.maas.aliyuncs.com）；
- 支持直接传入完整旁白文本，或从 Script.json 合并 narration；
- 支持 --script 逐场景合成：每场景单独生成 ``scene_<id>.<格式>``，
  用 ffprobe/mutagen 读取真实音频时长，输出 ``durations.json``（含每场景
  实际时长、剧本预期时长、累计时间轴偏移），并可选用 ffmpeg 拼接总音频；
- 支持配置音色 ID、语速、音量、音调，以及情绪 / 风格（instruction）；
- 默认输出 MP3，自动创建 ``output/<时间戳>/voice.mp3``；
- 完整的 API 异常、超时、下载失败处理；
- 输出生成耗时、音频路径与文件大小。

CLI 示例::

    python 01-voice/generate.py --text "你好，欢迎使用阿里云百炼。"
    python 01-voice/generate.py --script LossAversion/Script.json \
        --voice longanlingxin --instruction "用温柔自然的女性旁白，语速适中"

作为模块调用::

    from voice.generate import load_config, synthesize
    cfg = load_config()
    result = synthesize("你好", cfg)
    print(result.audio_path)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import project as project_io  # noqa: E402

try:
    import requests
except ImportError:  # pragma: no cover
    sys.stderr.write("缺少依赖 requests，请先执行: pip install -r voice/requirements.txt\n")
    raise

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]


logger = logging.getLogger("voice.generate")

# --------------------------------------------------------------------------- #
# 常量与默认值
# --------------------------------------------------------------------------- #
MODEL_NAME = "qwen-audio-3.0-tts-plus"
DEFAULT_VOICE = "longanlingxin"        # plus 系统音色：龙安灵心（女·知心温暖音）
DEFAULT_FORMAT = "mp3"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_RATE = 1.0                     # 语速 [0.5, 2.0]
DEFAULT_VOLUME = 50                    # 音量 [0, 100]
DEFAULT_PITCH = 1.0                    # 音调 [0.5, 2.0]

API_TIMEOUT = (10, 120)                # (连接超时, 读取超时) 秒
DOWNLOAD_TIMEOUT = 120                 # 音频下载超时 秒

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = _SCRIPT_DIR / "output"

_REGION_HOSTS = {
    "cn-beijing": "{ws}.cn-beijing.maas.aliyuncs.com",
    "ap-southeast-1": "{ws}.ap-southeast-1.maas.aliyuncs.com",
}

# 非实时语音合成服务路径（标准百炼网关与 Token Plan 网关完全一致）
TTS_PATH = "/api/v1/services/audio/tts/SpeechSynthesizer"


# --------------------------------------------------------------------------- #
# 异常体系
# --------------------------------------------------------------------------- #
class TTSError(Exception):
    """TTS 服务基础异常。"""


class ConfigError(TTSError):
    """配置缺失或非法。"""


class APIError(TTSError):
    """阿里云百炼接口异常（含超时）。"""


class DownloadError(TTSError):
    """音频下载或写盘异常。"""


# --------------------------------------------------------------------------- #
# 配置与结果数据类
# --------------------------------------------------------------------------- #
@dataclass
class TTSConfig:
    """语音合成配置。敏感字段（api_key）不会出现在快照中。"""

    api_key: str
    workspace_id: str = ""
    region: str = "cn-beijing"
    base_url: Optional[str] = None   # 如 Token Plan: https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
    model: str = MODEL_NAME
    voice: str = DEFAULT_VOICE
    audio_format: str = DEFAULT_FORMAT
    sample_rate: int = DEFAULT_SAMPLE_RATE
    rate: float = DEFAULT_RATE
    volume: int = DEFAULT_VOLUME
    pitch: float = DEFAULT_PITCH
    instruction: Optional[str] = None
    seed: Optional[int] = None
    language_hints: Optional[list] = None

    @property
    def endpoint(self) -> str:
        # 优先使用显式 Base URL（如 Token Plan 网关）：仅取 scheme+host，拼接原生 TTS 路径。
        # 注意：TTS 走 /api/v1/services/... 而非 /compatible-mode/v1，故此处只复用主机名。
        if self.base_url:
            parsed = urlparse(self.base_url.strip())
            if not parsed.scheme or not parsed.netloc:
                raise ConfigError(f"DASHSCOPE_BASE_URL 非法: {self.base_url!r}")
            return f"{parsed.scheme}://{parsed.netloc}{TTS_PATH}"
        # 否则用 WorkspaceId + region 拼接标准百炼网关
        if not self.workspace_id:
            raise ConfigError(
                "缺少接入点：请在 .env 配置 DASHSCOPE_BASE_URL（Token Plan 网关）"
                "或 DASHSCOPE_WORKSPACE_ID（标准百炼网关）"
            )
        tpl = _REGION_HOSTS.get(self.region)
        if not tpl:
            raise ConfigError(
                f"不支持的 region: {self.region!r}，可选: {list(_REGION_HOSTS)}"
            )
        host = tpl.format(ws=self.workspace_id)
        return f"https://{host}{TTS_PATH}"

    def safe_snapshot(self) -> dict:
        """返回脱敏后的配置快照（隐藏 api_key）。"""
        data = asdict(self)
        data.pop("api_key", None)
        if data.get("workspace_id"):
            data["workspace_id"] = _mask(data["workspace_id"])
        return data


@dataclass
class TTSResult:
    """一次合成任务的结果。"""

    output_dir: Path
    audio_path: Path
    file_size: int
    duration_seconds: float
    request_id: str
    audio_url: str
    characters: int
    config_snapshot: dict = field(default_factory=dict)


@dataclass
class _APIResponse:
    """一次 TTS 接口调用的原始产物（尚未写盘）。"""

    audio_bytes: bytes
    elapsed_seconds: float
    request_id: str
    audio_url: str
    characters: int


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
            return f"{num:.2f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024
    return f"{num:.2f} TB"


def _resolve_env_path(explicit: Optional[Path] = None) -> Optional[Path]:
    """按优先级查找 .env：显式参数 > voice/.env > 项目根/.env > cwd/.env。"""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None
    candidates = [
        _SCRIPT_DIR / ".env",
        _SCRIPT_DIR.parent / ".env",
        Path.cwd() / ".env",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """若目标文件已存在，则追加 _1/_2… 避免同秒覆盖。"""
    candidate = directory / f"{stem}{suffix}"
    idx = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{idx}{suffix}"
        idx += 1
    return candidate


def _format_api_error(resp: "requests.Response") -> str:
    """尽量从错误响应中解析 code/message，拼接可读错误。"""
    text = (resp.text or "")[:800]
    try:
        data = resp.json()
        code = data.get("code") or data.get("Code")
        message = data.get("message") or data.get("Message")
        request_id = data.get("request_id") or data.get("RequestId")
        if code or message:
            return f"HTTP {resp.status_code} | code={code} | message={message} | request_id={request_id}"
    except ValueError:
        pass
    return f"HTTP {resp.status_code} | {text}"


# --------------------------------------------------------------------------- #
# 配置加载
# --------------------------------------------------------------------------- #
def load_config(env_path: Optional[Path] = None, **overrides: Any) -> TTSConfig:
    """从 ``.env`` 加载配置，并用 ``overrides`` 中非 None 的值覆盖。

    环境变量（.env）：
        DASHSCOPE_API_KEY        必填，百炼 / Token Plan API Key
        DASHSCOPE_BASE_URL       接入点 Base URL（Token Plan 推荐），如
                                 https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
        DASHSCOPE_WORKSPACE_ID   业务空间 ID（标准百炼网关，与 BASE_URL 二选一）
        DASHSCOPE_REGION         可选，cn-beijing（默认）/ ap-southeast-1
        TTS_MODEL / TTS_VOICE / TTS_FORMAT / TTS_SAMPLE_RATE
        TTS_RATE / TTS_VOLUME / TTS_PITCH / TTS_INSTRUCTION
        TTS_SEED / TTS_LANGUAGE_HINTS
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
    workspace_id = (os.getenv("DASHSCOPE_WORKSPACE_ID") or "").strip()
    base_url = (os.getenv("DASHSCOPE_BASE_URL") or "").strip() or None
    if not api_key:
        raise ConfigError("缺少 DASHSCOPE_API_KEY，请在 .env 中配置（参考 voice/.env.example）")
    if not base_url and not workspace_id:
        raise ConfigError(
            "缺少接入点：请在 .env 配置 DASHSCOPE_BASE_URL（如 Token Plan 网关）"
            "或 DASHSCOPE_WORKSPACE_ID（标准百炼网关）"
        )

    def _env(name: str, default: Any) -> Any:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            return default
        return raw

    kwargs: dict = {
        "api_key": api_key,
        "workspace_id": workspace_id,
        "base_url": base_url,
        "region": str(_env("DASHSCOPE_REGION", "cn-beijing")).strip(),
        "model": str(_env("TTS_MODEL", MODEL_NAME)).strip(),
        "voice": str(_env("TTS_VOICE", DEFAULT_VOICE)).strip(),
        "audio_format": str(_env("TTS_FORMAT", DEFAULT_FORMAT)).strip().lower(),
        "sample_rate": int(_env("TTS_SAMPLE_RATE", DEFAULT_SAMPLE_RATE)),
        "rate": float(_env("TTS_RATE", DEFAULT_RATE)),
        "volume": int(_env("TTS_VOLUME", DEFAULT_VOLUME)),
        "pitch": float(_env("TTS_PITCH", DEFAULT_PITCH)),
        "instruction": (str(_env("TTS_INSTRUCTION", "")).strip() or None),
    }

    seed_raw = os.getenv("TTS_SEED")
    if seed_raw and str(seed_raw).strip():
        kwargs["seed"] = int(seed_raw)

    hints_raw = os.getenv("TTS_LANGUAGE_HINTS")
    if hints_raw and str(hints_raw).strip():
        kwargs["language_hints"] = [h.strip() for h in str(hints_raw).split(",") if h.strip()]

    # CLI / 调用方覆盖（仅覆盖显式提供的非 None 值）
    kwargs.update({k: v for k, v in overrides.items() if v is not None})

    cfg = TTSConfig(**kwargs)
    _validate(cfg)
    return cfg


def _validate(cfg: TTSConfig) -> None:
    if cfg.audio_format not in ("mp3", "wav", "pcm", "opus"):
        raise ConfigError(f"不支持的音频格式: {cfg.audio_format}")
    if not 0.5 <= cfg.rate <= 2.0:
        raise ConfigError(f"rate 需在 [0.5, 2.0]，当前 {cfg.rate}")
    if not 0 <= cfg.volume <= 100:
        raise ConfigError(f"volume 需在 [0, 100]，当前 {cfg.volume}")
    if not 0.5 <= cfg.pitch <= 2.0:
        raise ConfigError(f"pitch 需在 [0.5, 2.0]，当前 {cfg.pitch}")
    valid_rates = {8000, 12000, 16000, 22050, 24000, 44100, 48000}
    if cfg.sample_rate not in valid_rates:
        raise ConfigError(f"sample_rate 需为 {sorted(valid_rates)} 之一，当前 {cfg.sample_rate}")
    if cfg.audio_format == "opus" and cfg.sample_rate not in {8000, 12000, 16000, 24000, 48000}:
        raise ConfigError("format=opus 时 sample_rate 仅支持 8000/12000/16000/24000/48000")
    if cfg.seed is not None and not 0 <= cfg.seed <= 65535:
        raise ConfigError(f"seed 需在 [0, 65535]，当前 {cfg.seed}")


# --------------------------------------------------------------------------- #
# 核心合成逻辑
# --------------------------------------------------------------------------- #
def _build_payload(text: str, cfg: TTSConfig) -> dict:
    inp: dict = {
        "text": text,
        "voice": cfg.voice,
        "format": cfg.audio_format,
        "sample_rate": cfg.sample_rate,
        "rate": cfg.rate,
        "volume": cfg.volume,
        "pitch": cfg.pitch,
    }
    if cfg.instruction:
        inp["instruction"] = cfg.instruction
    if cfg.seed is not None:
        inp["seed"] = cfg.seed
    if cfg.language_hints:
        inp["language_hints"] = cfg.language_hints
    return {"model": cfg.model, "input": inp}


def _request_tts(text: str, cfg: TTSConfig, started: Optional[float] = None) -> _APIResponse:
    """调用百炼 TTS 接口并下载音频字节（不写盘）。

    Args:
        text: 待合成文本。
        cfg: 配置。
        started: 计时起点（time.perf_counter()），默认取当前时刻。

    Returns:
        _APIResponse: 含音频字节、耗时、request_id、URL、计费字符数。

    Raises:
        TTSError / APIError / DownloadError
    """
    text = (text or "").strip()
    if not text:
        raise TTSError("合成文本为空")

    payload = _build_payload(text, cfg)
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    if started is None:
        started = time.perf_counter()

    logger.info("→ 请求合成: model=%s voice=%s format=%s chars=%d",
                cfg.model, cfg.voice, cfg.audio_format, len(text))
    logger.debug("payload=%s", json.dumps(payload, ensure_ascii=False))

    # 1) 调用合成接口
    try:
        resp = requests.post(cfg.endpoint, headers=headers, json=payload, timeout=API_TIMEOUT)
    except requests.exceptions.Timeout as exc:
        raise APIError(f"调用百炼接口超时（>{API_TIMEOUT[1]}s）: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise APIError(f"调用百炼接口网络异常: {exc}") from exc

    if resp.status_code != 200:
        detail = _format_api_error(resp)
        if resp.status_code == 400 and (
            "Engine error [411]" in detail or "TTS speak operation failed" in detail
        ):
            detail += (
                "｜常见原因：voice 与 model 不匹配。qwen-audio-3.0-tts-plus 请用不带后缀的系统音色"
                "（longanlingxin / longanlufeng / longanhuan_v3.6 / longanlingxi）；"
                "带 _v3.1 后缀的音色属于 qwen-audio-3.1-tts-* 模型，Token Plan 不包含。"
            )
        raise APIError(f"百炼接口返回错误: {detail}")

    try:
        data = resp.json()
    except ValueError as exc:
        raise APIError(f"百炼响应非合法 JSON: {(resp.text or '')[:300]}") from exc

    audio = (data.get("output") or {}).get("audio") or {}
    audio_url = audio.get("url")
    request_id = data.get("request_id", "")
    characters = int((data.get("usage") or {}).get("characters", 0) or 0)
    if not audio_url:
        raise APIError(
            "百炼响应缺少 output.audio.url: "
            f"{json.dumps(data, ensure_ascii=False)[:400]}"
        )

    # 2) 下载音频
    try:
        audio_resp = requests.get(audio_url, timeout=DOWNLOAD_TIMEOUT)
        audio_resp.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise DownloadError(f"下载音频超时（>{DOWNLOAD_TIMEOUT}s）: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise DownloadError(f"下载音频失败: {exc}") from exc

    audio_bytes = audio_resp.content
    if not audio_bytes:
        raise DownloadError("下载到的音频内容为空（0 字节）")

    return _APIResponse(
        audio_bytes=audio_bytes,
        elapsed_seconds=round(time.perf_counter() - started, 2),
        request_id=request_id,
        audio_url=audio_url,
        characters=characters,
    )


def synthesize(
    text: str,
    cfg: TTSConfig,
    output_root: Optional[Path] = None,
) -> TTSResult:
    """调用百炼 HTTP API 合成语音并下载到带时间戳目录。

    Args:
        text: 完整旁白文本（可内嵌 [excited]/[sad] 等情感与富语言标签）。
        cfg: 由 :func:`load_config` 得到的配置。
        output_root: 输出根目录，默认 ``voice/output``。

    Returns:
        TTSResult: 含音频路径、文件大小、耗时、request_id 等。

    Raises:
        TTSError / APIError / DownloadError
    """
    root = Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
    resp_data = _request_tts(text, cfg)

    # 写入带时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / timestamp
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = _unique_path(out_dir, "voice", f".{cfg.audio_format}")
        audio_path.write_bytes(resp_data.audio_bytes)
    except OSError as exc:
        raise DownloadError(f"写入音频文件失败: {exc}") from exc

    result = TTSResult(
        output_dir=out_dir,
        audio_path=audio_path,
        file_size=len(resp_data.audio_bytes),
        duration_seconds=resp_data.elapsed_seconds,
        request_id=resp_data.request_id,
        audio_url=resp_data.audio_url,
        characters=resp_data.characters,
        config_snapshot=cfg.safe_snapshot(),
    )
    logger.info("✓ 合成完成: %s (%s, %.2fs)",
                result.audio_path, _human_size(result.file_size), result.duration_seconds)
    return result


# --------------------------------------------------------------------------- #
# 音频时长探测与拼接
# --------------------------------------------------------------------------- #
def probe_audio_duration(path: Path) -> Optional[float]:
    """读取音频文件真实时长（秒）。

    探测顺序：ffprobe > mutagen > 按码率估算（仅 mp3/opus，误差较大）。
    全部失败时返回 None（不中断合成流程）。
    """
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0 and out.stdout.strip():
                return round(float(out.stdout.strip()), 3)
            logger.warning("ffprobe 读取时长失败: %s", (out.stderr or "").strip()[:200])
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            logger.warning("ffprobe 异常（%s），尝试备用方案", exc)
    try:
        from mutagen import File as MutagenFile  # 延迟导入，未安装不影响主流程
        info = MutagenFile(str(path))
        if info is not None and getattr(info, "info", None) is not None:
            return round(float(info.info.length), 3)
    except ImportError:
        logger.warning("未安装 ffprobe 与 mutagen，无法精确读取时长；"
                       "建议 brew install ffmpeg 或 pip install mutagen")
    except Exception as exc:  # noqa: BLE001
        logger.warning("mutagen 读取时长失败: %s", exc)
    # 粗略估算（mp3 常见 48kbps / opus 常见 32kbps），仅作兜底
    suffix = path.suffix.lower()
    if suffix in (".mp3", ".opus"):
        bitrate = 48000 if suffix == ".mp3" else 32000
        estimated = path.stat().st_size * 8 / bitrate
        logger.warning("按时长估算兜底（误差较大）: %s ≈ %.2fs", path.name, estimated)
        return round(estimated, 2)
    return None


# --------------------------------------------------------------------------- #
# 逐 shot 多角色配音（projects/<id>/script.json）
# --------------------------------------------------------------------------- #
def resolve_ffmpeg(explicit: Optional[str] = None) -> Optional[str]:
    """解析 ffmpeg（拼接 shot 内多段音频用）；找不到返回 None。"""
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return str(p.resolve())
        found = shutil.which(explicit)
        if found:
            return found
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
    return shutil.which("ffmpeg")


def _concat_audio(ffmpeg_bin: str, seg_paths: list, out_path: Path, gap: float) -> None:
    """把多段音频按序拼接为 out_path（重编码统一参数），非末段尾部插入 gap 秒静音。"""
    if len(seg_paths) == 1:
        shutil.copyfile(seg_paths[0], out_path)
        return
    argv = [ffmpeg_bin, "-y"]
    for sp in seg_paths:
        argv += ["-i", str(sp)]
    n = len(seg_paths)
    parts = []
    for i in range(n):
        pad = f",apad=pad_dur={gap:.3f}" if (gap > 0 and i < n - 1) else ""
        parts.append(f"[{i}:a]aresample=24000,aformat=sample_fmts=fltp:channel_layouts=mono{pad}[a{i}]")
    concat_in = "".join(f"[a{i}]" for i in range(n))
    parts.append(f"{concat_in}concat=n={n}:v=0:a=1[out]")
    argv += ["-filter_complex", ";".join(parts), "-map", "[out]",
             "-c:a", "libmp3lame", "-q:a", "4", str(out_path)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not out_path.exists():
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise TTSError(f"拼接 shot 音频失败 (rc={proc.returncode}):\n  " + "\n  ".join(tail))


def _make_silence(ffmpeg_bin: str, out_path: Path, duration: float) -> None:
    """生成 duration 秒的静音 mp3（纯视觉镜头用，保证音视频 1:1 对齐）。"""
    argv = [ffmpeg_bin, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", f"{max(duration, 0.05):.3f}", "-c:a", "libmp3lame", "-q:a", "4", str(out_path)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0 or not out_path.exists():
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise TTSError(f"生成静音失败 (rc={proc.returncode}):\n  " + "\n  ".join(tail))


def synthesize_project(project_dir: Path, cfg: TTSConfig, *,
                       narrator_voice: Optional[str] = None, resume: bool = True,
                       gap: float = 0.10) -> dict:
    """逐 shot 多角色配音：narration(旁白音色) + dialogue(角色音色) → audio/<shot>.mp3。

    shot 真实时长 = 其音频时长（无旁白/对白的纯视觉 shot 用名义 duration_sec）；按 shot
    顺序累加得绝对时间轴；回写 script.json 的 runtime_sec/start/end/status/outputs。
    """
    project_dir = Path(project_dir).expanduser()
    script = project_io.load_script(project_dir)
    audio_dir = project_io.audio_dir(project_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = resolve_ffmpeg(None)
    narr_voice = narrator_voice or cfg.voice

    started = time.perf_counter()
    cursor = 0.0
    n_audio = n_silent = 0
    for scene, shot in script.iter_shots():
        out_path = project_io.audio_path(project_dir, shot.id)
        segs: list = []
        if shot.narration.strip():
            segs.append((shot.narration.strip(), narr_voice))
        for line in shot.dialogue:
            if not line.text.strip():
                continue
            char = script.characters.get(line.speaker)
            segs.append((line.text.strip(), char.voice if (char and char.voice) else narr_voice))

        if resume and out_path.exists() and out_path.stat().st_size > 0:
            runtime = probe_audio_duration(out_path)
            logger.info("→ shot %s 已有音频，跳过（%.2fs）", shot.id, runtime or 0.0)
        elif not segs:
            if ffmpeg_bin is None:
                raise TTSError("生成静音需要 ffmpeg，请 pip install imageio-ffmpeg")
            _make_silence(ffmpeg_bin, out_path, float(shot.duration_sec))
            runtime = probe_audio_duration(out_path) or float(shot.duration_sec)
            n_silent += 1
            logger.info("→ shot %s 纯视觉，生成 %.2fs 静音", shot.id, runtime)
        else:
            if len(segs) > 1 and ffmpeg_bin is None:
                raise TTSError("多段音频拼接需要 ffmpeg，请 pip install imageio-ffmpeg")
            tmpdir = Path(tempfile.mkdtemp(prefix="vf_tts_"))
            try:
                seg_paths = []
                for i, (text, voice) in enumerate(segs):
                    resp = _request_tts(text, replace(cfg, voice=voice), started=started)
                    sp = tmpdir / f"seg_{i:02d}.mp3"
                    sp.write_bytes(resp.audio_bytes)
                    seg_paths.append(sp)
                if len(seg_paths) == 1:
                    shutil.copyfile(seg_paths[0], out_path)
                else:
                    _concat_audio(ffmpeg_bin, seg_paths, out_path, gap)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
            runtime = probe_audio_duration(out_path)
            if runtime is None:
                raise TTSError(f"shot {shot.id} 无法获取音频时长（请安装 mutagen 或 ffmpeg）")
            n_audio += 1
            logger.info("✓ shot %s: %d 段 → %.2fs", shot.id, len(segs), runtime)

        if runtime is None:
            runtime = float(shot.duration_sec)
        shot.runtime_sec = round(runtime, 3)
        shot.start = round(cursor, 3)
        cursor += runtime
        shot.end = round(cursor, 3)
        shot.status.audio = "done"
        shot.outputs.audio = project_io.rel(project_dir, out_path)

    project_io.save_script(script, project_dir / "script.json")

    errors = []
    prev = 0.0
    for s in script.all_shots():
        if s.start is None or s.end is None or s.end <= s.start:
            errors.append(f"shot {s.id} 时间轴非法")
            continue
        if abs(s.start - prev) > 1e-3:
            errors.append(f"shot {s.id} start={s.start} 与上一 end={prev} 不连续")
        prev = s.end
    if errors:
        for e in errors:
            logger.error("  [error] %s", e)
        raise TTSError(f"时间轴校验失败（{len(errors)} 处）")

    return {"project_dir": project_dir, "audio_dir": audio_dir,
            "shot_count": len(script.all_shots()), "audio_shots": n_audio,
            "silent_shots": n_silent, "total_seconds": round(prev, 3),
            "elapsed_seconds": round(time.perf_counter() - started, 2)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice.generate",
        description="阿里云百炼 qwen-audio-3.0-tts-plus 语音合成服务",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="直接传入完整旁白文本")
    src.add_argument("--text-file", help="从文本文件读取旁白")
    src.add_argument("--project", help="短剧项目目录（projects/<id>）：读 script.json，逐 shot 多角色配音 → audio/ + 回写时间轴")

    parser.add_argument("--voice", help=f"音色 ID（默认 {DEFAULT_VOICE}）")
    parser.add_argument("--rate", type=float, help="语速 0.5-2.0")
    parser.add_argument("--volume", type=int, help="音量 0-100")
    parser.add_argument("--pitch", type=float, help="音调 0.5-2.0")
    parser.add_argument("--instruction", help="情绪/风格/方言自然语言指令，如：用温柔自然的女性旁白")
    parser.add_argument("--format", dest="audio_format",
                        choices=["mp3", "wav", "pcm", "opus"], help="音频格式")
    parser.add_argument("--sample-rate", type=int, help="采样率 Hz")
    parser.add_argument("--seed", type=int, help="随机种子 0-65535")
    parser.add_argument("--output", help=f"输出根目录（默认 {DEFAULT_OUTPUT_ROOT}）")
    parser.add_argument("--no-resume", action="store_true",
                        help="--project 模式下不跳过已存在的 shot 音频，强制重新合成")
    parser.add_argument("--narrator-voice", help="旁白音色 ID（默认取 .env 的 TTS_VOICE）")
    parser.add_argument("--gap", type=float, default=0.10,
                        help="shot 内多段音频（旁白/对白）之间的静音间隔秒")
    parser.add_argument("--env", help="指定 .env 文件路径")
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
        # 1) 加载配置（CLI 覆盖 .env）
        cfg = load_config(
            env_path=Path(args.env) if args.env else None,
            voice=args.voice,
            rate=args.rate,
            volume=args.volume,
            pitch=args.pitch,
            instruction=args.instruction,
            audio_format=args.audio_format,
            sample_rate=args.sample_rate,
            seed=args.seed,
        )

        # 2) --project：短剧流水线（逐 shot 多角色配音 + 时间轴回写）
        if args.project:
            res = synthesize_project(Path(args.project).expanduser(), cfg,
                                     narrator_voice=args.narrator_voice,
                                     resume=not args.no_resume, gap=args.gap)
            logger.info("— 结果 —")
            logger.info("  镜头总数  : %d（有声 %d / 纯视觉 %d）",
                        res["shot_count"], res["audio_shots"], res["silent_shots"])
            logger.info("  总时长    : %.3fs", res["total_seconds"])
            logger.info("  耗时      : %.1fs", res["elapsed_seconds"])
            logger.info("  音频目录  : %s", res["audio_dir"].resolve())
            logger.info("  已回写    : %s", (res["project_dir"] / "script.json").resolve())
            print(json.dumps({
                "ok": True, "mode": "project",
                "project_dir": str(res["project_dir"].resolve()),
                "audio_dir": str(res["audio_dir"].resolve()),
                "shot_count": res["shot_count"], "audio_shots": res["audio_shots"],
                "silent_shots": res["silent_shots"], "total_seconds": res["total_seconds"],
            }, ensure_ascii=False))
            return 0

        # 3) --text / --text-file：整段合成
        if args.text:
            text = args.text
        else:
            tf = Path(args.text_file).expanduser()
            if not tf.exists():
                raise TTSError(f"文本文件不存在: {tf}")
            text = tf.read_text(encoding="utf-8").strip()
        if not text:
            raise TTSError("未获得任何待合成文本")

        # 4) 合成
        result = synthesize(text, cfg, output_root=Path(args.output) if args.output else None)

        # 5) 输出耗时 / 路径 / 大小
        logger.info("— 结果 —")
        logger.info("  耗时      : %.2fs", result.duration_seconds)
        logger.info("  音频路径  : %s", result.audio_path.resolve())
        logger.info("  文件大小  : %s (%d bytes)", _human_size(result.file_size), result.file_size)
        logger.info("  request_id: %s", result.request_id)
        logger.info("  计费字符  : %d", result.characters)
        logger.info("  音频 URL  : %s", result.audio_url)

        # 机器可读结果（便于上层流水线消费）
        print(json.dumps({
            "ok": True,
            "audio_path": str(result.audio_path.resolve()),
            "output_dir": str(result.output_dir.resolve()),
            "file_size": result.file_size,
            "duration_seconds": result.duration_seconds,
            "request_id": result.request_id,
            "characters": result.characters,
            "audio_url": result.audio_url,
        }, ensure_ascii=False))
        return 0

    except ConfigError as exc:
        logger.error("配置错误: %s", exc)
        return 3
    except TTSError as exc:
        logger.error("合成失败: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("用户中断")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("未预期错误: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
