# x-tools 视频处理工具箱

一个简单易用的批处理工具箱，专注于视频修复与增强。

**核心功能**: 去水印 (OpenCV/LaMA 深度学习)、高清重置 (Real-ESRGAN/FFmpeg)、关键帧提取、内容截取、帧数补充 (RIFE)。

## � 快速开始 (30秒上手)

**1. 环境配置**
需要 Python 3.10+ 和 FFmpeg。

```bash
# 创建并激活环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

**2. 运行交互式终端**
无需记忆命令，通过箭头键选择功能：

```bash
python main.py
```

支持：
- 📂 **输入源灵活**: 自动扫描 `input/` 目录，或选择单个文件/任意文件夹。
- 💧 **去水印**: 支持鼠标框选区域 (OpenCV 快速修复 / LaMA 深度学习无痕修复)。
- 🆙 **高清重置**: 批量 2x/4x 放大 (使用 Real-ESRGAN AI 或 FFmpeg)。
- ✂️ **内容提取**: 批量截取片段、提取关键帧。

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

### 2. 🆙 视频高清重置 (超分辨率)
```bash
# Real-ESRGAN AI 超分 (2倍放大)
python tools/upscale/realesrgan.py video.mp4 -s 2

# FFmpeg 传统放大 (速度快)
python tools/upscale/ffmpeg_scale.py video.mp4 -s 2
```

### 3. ✂️ 内容截取与提取
```bash
# 截取前 30 秒
python tools/extract/clip_extractor.py video.mp4 -s 0 -d 30

# 提取所有关键帧 (I-Frame)
python tools/extract/keyframe_extractor.py video.mp4 --keyframes
```

### 4. ⏯️ 帧数补充 (插帧)
```bash
# RIFE AI 插帧 (2倍帧率)
python tools/interpolation/rife.py video.mp4 -m 2
```

---

## �️ 目录结构
```
x-tools/
├── main.py                       # 🚀 交互式入口
├── config.py                     # ⚙️ 全局配置
├── input/                        # 📂 默认输入目录
├── output/                       # 📂 默认输出目录
└── tools/
    ├── watermark/                # 去水印模块 (OpenCV, LaMA)
    ├── upscale/                  # 超分模块 (Real-ESRGAN, FFmpeg)
    ├── extract/                  # 提取模块 (截取, 关键帧)
    └── interpolation/            # 插帧模块 (RIFE, FFmpeg)
```

## 📦 依赖说明
- **基础依赖**: `opencv-python`, `ffmpeg-python`, `rich`, `InquirerPy`
- **AI 增强 (按需安装)**:
  - 去水印 (LaMA): `torch`, `torchvision` (首次运行自动下载模型)
  - 超分 (Real-ESRGAN): `basicsr`, `realesrgan`
  - 插帧 (RIFE): `rife-ncnn-vulkan-python`