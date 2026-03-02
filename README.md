# x-tools 视频处理工具箱

一个简单易用的批处理工具箱，专注于视频修复与增强。

**核心功能**: 去水印 (OpenCV/LaMA 深度学习)、高清重置 (Real-ESRGAN/FFmpeg)、帧数补充 (RIFE)、格式转换 (FFmpeg)。

## � 快速开始 (30秒上手)

**1. 安装 FFmpeg**

| 平台 | 命令 |
|------|------|
| macOS | `brew install ffmpeg` |
| Ubuntu | `sudo apt install ffmpeg` |
| Windows | `winget install Gyan.FFmpeg` (安装后需**重启终端**) |

**2. 环境配置**

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> ⚠️ **Windows 注意**: 请使用 `python` 而非 `python3`，激活虚拟环境时必须使用 `& .\.venv\Scripts\Activate.ps1` 语法。
> 若安装 FFmpeg 后仍提示未检测到，请重启终端或执行:
> ```powershell
> $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
> ```

**3. 运行交互式终端**
无需记忆命令，通过箭头键选择功能：

```bash
python main.py
```

支持：
- 📂 **输入源灵活**: 自动扫描 `input/` 目录，或选择单个文件/任意文件夹。
- 💧 **去水印**: 支持鼠标框选区域 (OpenCV 快速修复 / LaMA 深度学习无痕修复)。
- 🏷️ **加水印**: 文字水印 (支持中文) / 图片水印 (Logo), 支持图片和视频。
- 🆙 **高清重置**: 批量 2x/4x 放大 (使用 Real-ESRGAN AI 或 FFmpeg)。
- 🔄 **格式转换**: 视频格式互转 / 提取音频 / 去除音频 / 快速无损封装。
- 📊 **查看信息**: 显示分辨率、清晰度等级、帧率、编码器、码率、音频等详细信息。

---

## 📚 命令行工具 (高级用法)

如果你更喜欢 CLI 或需要集成到脚本中：

### 1. 💧 视频去水印
支持指定坐标 `x1,y1,x2,y2` 或 mask 图片。

```bash
# OpenCV 快速模式 (坐标: 10,10,200,60)
python tools/watermark/opencv_inpaint.py video.mp4 -r 10,10,200,60

# LaMA 深度学习模式 (效果最好, 自动下载模型)
python tools/watermark/lama_remover.py video.mp4 -r 10,10,200,60

# 批量处理 input/ 下所有视频
python tools/watermark/batch.py -r 625,1220,695,1260 lama
```

### 2. 🏷️ 增加水印
支持文字水印 (中文) 和图片水印 (Logo)，同时支持图片和视频。

```bash
# 文字水印 (图片或视频均可)
python tools/add_watermark/text_watermark.py image.jpg -t "© 2026 版权所有"
python tools/add_watermark/text_watermark.py video.mp4 -t "Sample" --position top-left --opacity 0.5

# Logo 水印
python tools/add_watermark/image_watermark.py video.mp4 -w logo.png --scale 0.2

# 批量处理
python tools/add_watermark/batch.py text -t "水印文字"
python tools/add_watermark/batch.py image -w logo.png
```

### 3. 🆙 视频高清重置 (超分辨率)
```bash
# Real-ESRGAN AI 超分 (2倍放大)
python tools/upscale/realesrgan.py video.mp4 -s 2

# FFmpeg 传统放大 (速度快)
python tools/upscale/ffmpeg_scale.py video.mp4 -s 2
```

### 4. ⏯️ 帧数补充 (插帧)
```bash
# RIFE AI 插帧 (2倍帧率)
python tools/interpolation/rife.py video.mp4 -m 2
```

### 5. 🔄 格式转换
```bash
# 视频格式转换 (MKV → MP4)
python tools/convert/ffmpeg_convert.py video.mkv -f mp4

# 转码为 H.265 (更好压缩)
python tools/convert/ffmpeg_convert.py video.mp4 -f mp4 --video-codec libx265

# 提取音频 (视频 → MP3)
python tools/convert/ffmpeg_convert.py video.mp4 -f mp3

# 去除音频 (仅保留视频)
python tools/convert/ffmpeg_convert.py video.mp4 -f mp4 --strip-audio

# 快速封装 (无损换容器, 极快)
python tools/convert/ffmpeg_convert.py video.mkv -f mp4 --copy

# 批量转换
python tools/convert/batch.py -f mp4
```

### 6. 📊 查看媒体信息
```bash
# 单个文件详细信息 (分辨率、清晰度、帧率、编码器、码率、音频等)
python tools/mediainfo/probe.py video.mp4

# 多文件汇总对比
python tools/mediainfo/probe.py video1.mp4 video2.mkv video3.mov
```

---

## 🗂️ 目录结构
```
x-tools/
├── main.py                       # 🚀 交互式入口
├── config.py                     # ⚙️ 全局配置
├── input/                        # 📂 默认输入目录
├── output/                       # 📂 默认输出目录
└── tools/
    ├── watermark/                # 去水印模块 (OpenCV, LaMA)
    ├── add_watermark/            # 加水印模块 (文字, Logo)
    ├── upscale/                  # 超分模块 (Real-ESRGAN, FFmpeg)
    ├── interpolation/            # 插帧模块 (RIFE, FFmpeg)
    ├── convert/                  # 格式转换模块 (FFmpeg)
    └── mediainfo/                # 媒体信息查看 (FFprobe)
```

## 📦 依赖说明
- **基础依赖**: `opencv-python`, `ffmpeg-python`, `rich`, `InquirerPy`, `Pillow`
- **AI 增强 (按需安装)**:
  - 去水印 (LaMA): `torch`, `torchvision` (首次运行自动下载模型)
  - 超分 (Real-ESRGAN): `basicsr`, `realesrgan`
  - 插帧 (RIFE): `rife-ncnn-vulkan-python`