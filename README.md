# Video Factory

把「一段话」变成竖屏 AI 短剧的自动化流水线：LLM 写分镜脚本 → 逐镜头多角色配音 → 角色定妆照 + 关键帧 →
wan3.0-video 图生视频 → 字幕 → 合成。以 `script_template.json` 为数据模型，**script.json 是唯一数据源**，
提示词由代码按模板拼接，角色一致性靠定妆照 + 多图参考。

## 一键运行

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r common/requirements.txt -r 00-scripts/requirements.txt \
            -r 01-voice/requirements.txt -r 02-image/requirements.txt -r 03-video/requirements.txt
# 在 00-scripts / 01-voice / 02-image / 03-video 各目录下配置 .env（参考各自 .env.example）

# 全量（含 AI 视频）
python run.py --text "一个关于坚持与告别的短剧" --project-id the-last-track --target-duration 70

# 省钱快跑：跳过 AI 视频（所有镜头用关键帧 Ken Burns 运镜）
python run.py --text "..." --target-duration 30 --skip-ai

# 断点续跑：中断后用同一个项目目录继续（已生成的音频/图片/AI片段会跳过）
python run.py --project projects/the-last-track
```

所有产物写入统一项目目录 `projects/<project-id>/`。

## 模块与数据流

| 步骤 | 模块 | 职责 | 项目目录内 输入 → 输出 |
|-|-|-|-|
| 0 | `00-scripts` | 一段话 → 分镜脚本（qwen3.6-plus，Pydantic 校验 + 自动修复） | `--text` → `script.json` |
| 1 | `01-voice` | 逐 shot 多角色配音（旁白 + 对白），测真实时长并回写时间轴 | `script.json` → `audio/<shot>.mp3` |
| 2 | `02-image` | 角色定妆照 + 逐 shot 关键帧（wan2.7 多图参考保一致性） | `script.json` → `assets/chars/`、`keyframes/` |
| 3 | `03-video` | wan3.0-video 图生视频 + 逐 shot 定长装配 | `keyframes/` → `video/ai_<shot>.mp4`、`video/<shot>.mp4` |
| 4 | `04-subtitle` | 旁白 + 对白字幕（对白带「角色名：」前缀） | `script.json` → `subtitle.srt` |
| 5 | `05-compose` | 逐 shot 音视频 1:1 拼接 + 混音 + 烧字幕 | → `final.mp4` |

公共库 `common/`：`schema.py`（Pydantic 数据模型）、`prompts.py`（模板拼接提示词）、`project.py`（项目 IO 与状态回写）。

## 项目目录结构

```
projects/<project-id>/
  script.json          # 唯一数据源：project/characters/scenes[].shots[] + prompt_templates + status/outputs
  assets/chars/<key>.png   # 角色定妆照（02-image 生成，作关键帧的多图参考）
  keyframes/<shot>.png     # 每 shot 关键帧（首帧）
  audio/<shot>.mp3         # 每 shot 音频（旁白+对白；纯视觉镜头为静音）
  video/ai_<shot>.mp4      # wan3.0-video 图生视频原片
  video/<shot>.mp4         # 定长装配段
  subtitle.srt
  final.mp4
```

## 关键设计

- **script.json 是唯一数据源**：镜头只存结构化字段（`shot_type`/`characters`/`action`/`motion`/`narration`/
  `dialogue`/`duration_sec`），提示词由 `common/prompts.py` 按 `prompt_templates` 拼接，不手写进镜头。
- **角色一致性**：`characters` 里每个角色有固定 `appearance`/`outfit`（原样复用）；`02-image` 先生成定妆照，
  再以「定妆照(base64) + 关键帧提示词」走 wan2.7 多图参考生成关键帧；`video` 镜头以关键帧作 wan3.0-video 首帧。
- **时间轴天然精确**：`01-voice` 逐 shot 配音，shot 真实时长 = 其音频时长（`runtime_sec`），按顺序累加得到
  绝对时间轴并回写 script.json；纯视觉镜头生成等长静音，保证音视频 1:1。
- **多角色对白**：每个 shot 的 `narration`（旁白音色）+ `dialogue[]`（各角色 `voice`）逐条 TTS 后拼接。
- **status/outputs 回写**：各模块把 `status.{keyframe,video,audio}` 与 `outputs.*` 写回 script.json，支持断点续跑。
- **失败回退**：AI 视频失败的镜头，装配阶段自动回退为关键帧 Ken Burns 运镜，不中断流水线。

## 分步运行（不用 run.py）

```bash
P=projects/the-last-track
python 00-scripts/generate.py --text "..." --project $P --project-id the-last-track --target-duration 70
python 01-voice/generate.py    --project $P
python 02-image/generate.py    --project $P          # --stage chars|keyframes|all
python 03-video/generate.py    --project $P          # --stage ai|assemble|all
python 04-subtitle/generate.py --project $P
python 05-compose/generate.py  --project $P          # 可加 --bgm assets/bgm.mp3 --duck
```

## 说明

- **模型名**：Token Plan 网关不提供通用名 `qwen-plus`，脚本生成默认用 `qwen3.6-plus`
  （可选 `qwen3.7-plus`/`qwen4-max` 等，见 `00-scripts/.env`）。可用模型可 `GET {base_url}/models` 查询。
- **成本**：一集约 N 张关键帧 + 角色定妆照 + （默认）每个有 motion 的镜头一个 AI 视频（wan3.0-video 每任务约 1~5 分钟）。
  调试可加 `--skip-ai`、`--target-duration 30`、`--ai-resolution 480P`、`--stop-after image` 降本。
- **依赖**：`pydantic>=2`（schema 校验）、`requests`、`python-dotenv`、`imageio-ffmpeg`（ffmpeg 二进制）、`mutagen`（音频时长）。
- 数据模型定义见根目录 `script_template.json` 与 `common/schema.py`。

## Tips
```python
python3 -m venv .venv
```
