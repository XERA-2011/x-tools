# x-tools 视频处理工具箱

`x-tools` 是一个基于终端交互菜单（TUI）的媒体批处理工具，聚焦视频常见处理流程。

## 快速开始

1. 安装 FFmpeg
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt install ffmpeg`
- Windows: `winget install Gyan.FFmpeg`

2. 安装 Python 依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. 启动

```bash
python main.py
```

## 功能概览

- 去水印：FFmpeg delogo / OpenCV / LaMA
- 增加水印：文字水印、图片水印（支持图片与视频）
- 无损压缩：H.264 / H.265（CRF 预设）
- 高清重置：FFmpeg 放大、Real-ESRGAN 超分
- 帧数补充：FFmpeg 运动补偿、RIFE AI 插帧
- 格式转换：转码、提取音频、去音频、快速封装
- 滤镜效果：内置多种 FFmpeg 预设
- 裁切比例：1:1、3:4、4:3、9:16、16:9
- 拼接视频：支持转场和音频过渡
- 幻灯片：支持自定义模式与"热点批量模式"（按子目录一键批量出片，适合新闻/娱乐/科技热点）
- 添加背景音乐：支持混音与淡入淡出
- 字幕：自动识别、烧录、一键字幕、基于 SRT 的 AI 配音
- 歌词 MV 生成：基于音频节拍智能生成带排版的高燃文字向音乐视频 (Lyric MV)
- PDF 拆分：按页拆分、按范围拆分、按固定页数拆分、提取指定页
- 清理文件：清理 `input/` 和 `output/`
- 自检测试：一键验证环境与核心函数

## 项目结构

```
x-tools/
├── main.py              # TUI 主入口 (菜单分发)
├── config.py            # 全局配置、路径、常量
├── menus/               # TUI 交互菜单 (每个功能一个文件)
│   ├── _prompts.py      # 公共交互组件
│   ├── watermark.py     # 去水印
│   ├── subtitle.py      # 字幕
│   └── ...
├── tools/               # 核心处理逻辑
│   ├── common.py        # 批量调度、进度条、工具函数
│   ├── ffmpeg/          # FFmpeg 滤镜图构建
│   ├── watermark/       # 去水印引擎
│   ├── subtitle/        # 字幕识别与烧录
│   └── ...
├── tests/               # 单元测试
├── input/               # 默认输入目录
├── output/              # 默认输出目录
└── music/               # 背景音乐目录
```

## 测试

```bash
# 命令行运行
python -m pytest tests/ -v

# 或在 TUI 菜单中选择「🩺 自检测试」
python main.py
```

## 可选 AI 依赖（按需安装）

- LaMA 去水印：`pip install torch torchvision`
- Whisper 字幕识别：`pip install openai-whisper`
- RIFE 插帧：`pip install rife-ncnn-vulkan-python`
- 字幕配音（TTS）：`pip install edge-tts`
- Real-ESRGAN 超分：将 `realesrgan-ncnn-vulkan` 可执行文件放到 `bin/` 目录并赋予执行权限

## 说明

- 推荐 Python 3.10+
- macOS 如遇字幕烧录问题，可安装 `ffmpeg-full`（带 `libass`）
