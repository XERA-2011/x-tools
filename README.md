# x-tools

视频处理实验工具箱 — 基于 Python + FFmpeg

## ✨ 功能模块

| 模块 | 功能 | 状态 |
|------|------|------|
| `tools/extract/` | 视频片段截取 / 关键帧提取 | ✅ 可用 |
| `tools/watermark/` | 视频去水印 | ✅ 可用 |
| `tools/upscale/` | 视频高清重置 (超分辨率) | ✅ 可用 |
| `tools/interpolation/` | 视频帧数补充 (插帧) | ✅ 可用 |

## 🛠️ 环境配置

> 需要: Python 3.10+, FFmpeg

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 退出环境
deactivate
```

确认 FFmpeg 已安装:
```bash
brew install ffmpeg   # macOS
ffmpeg -version
```

## 🚀 快速使用

### 视频片段截取
```bash
# 单文件: 截取 00:01:00 到 00:02:30
python tools/extract/clip_extractor.py video.mp4 -s 00:01:00 -e 00:02:30

# 单文件: 从 10 秒开始截取 30 秒
python tools/extract/clip_extractor.py video.mp4 -s 10 -d 30

# 批量: 对 input/ 下所有视频截取前 60 秒
python tools/extract/batch.py -i input clip -s 0 -d 60
```

### 关键帧提取
```bash
# 提取 I-帧 (关键帧)
python tools/extract/keyframe_extractor.py video.mp4 --keyframes

# 每 2 秒提取一帧
python tools/extract/keyframe_extractor.py video.mp4 --interval 2

# 按场景切换提取 (阈值 0.3)
python tools/extract/keyframe_extractor.py video.mp4 --scene 0.3

# 批量: 提取所有视频的关键帧
python tools/extract/batch.py -i input keyframes
```

### 视频去水印
```bash
# 单文件: 指定水印区域 (x1,y1,x2,y2)
python tools/watermark/opencv_inpaint.py video.mp4 -r 10,10,200,60

# 多个水印区域
python tools/watermark/opencv_inpaint.py video.mp4 -r 10,10,200,60 -r 500,10,700,60

# 使用 mask 图片 (白色=水印)
python tools/watermark/opencv_inpaint.py video.mp4 -m mask.png

# LaMA 深度学习 (需额外安装: pip install iopaint torch torchvision)
python tools/watermark/lama_remover.py video.mp4 -r 10,10,200,60

# 批量: 对 input/ 下所有视频去除相同位置的水印
python tools/watermark/batch.py -r 10,10,200,60 opencv
python tools/watermark/batch.py -r 10,10,200,60 lama
```

### 视频高清重置
```bash
# FFmpeg 传统放大 2x (lanczos 插值)
python tools/upscale/ffmpeg_scale.py video.mp4 -s 2

# FFmpeg 放大到指定分辨率
python tools/upscale/ffmpeg_scale.py video.mp4 -W 1920

# Real-ESRGAN AI 超分 (需安装: pip install realesrgan torch torchvision basicsr)
python tools/upscale/realesrgan.py video.mp4 -s 2

# 批量放大
python tools/upscale/batch.py ffmpeg -s 2
python tools/upscale/batch.py realesrgan -s 2
```

### 视频帧数补充
```bash
# FFmpeg 运动补偿插帧 (24fps → 60fps)
python tools/interpolation/ffmpeg_minterp.py video.mp4 -t 60

# FFmpeg 插帧 - blend 模式 (更快但有残影)
python tools/interpolation/ffmpeg_minterp.py video.mp4 -t 60 --mode blend

# RIFE AI 插帧 (需安装: pip install rife-ncnn-vulkan-python)
python tools/interpolation/rife.py video.mp4 -m 2

# 批量插帧
python tools/interpolation/batch.py ffmpeg -t 60
python tools/interpolation/batch.py rife -m 2
```

## 📁 目录结构

```
x-tools/
├── config.py                     # 全局配置
├── tools/
│   ├── common.py                 # 公共工具 (批量调度、日志)
│   ├── extract/                  # 内容截取
│   │   ├── clip_extractor.py     #   视频片段截取
│   │   ├── keyframe_extractor.py #   关键帧提取
│   │   └── batch.py              #   批量截取入口
│   ├── watermark/                # 去水印
│   │   ├── opencv_inpaint.py     #   OpenCV 传统修复
│   │   ├── lama_remover.py       #   LaMA 深度学习修复
│   │   └── batch.py              #   批量去水印入口
│   ├── upscale/                  # 高清重置
│   │   ├── realesrgan.py         #   Real-ESRGAN AI 超分
│   │   ├── ffmpeg_scale.py       #   FFmpeg 传统放大
│   │   └── batch.py              #   批量高清重置入口
│   └── interpolation/            # 帧数补充
│       ├── rife.py               #   RIFE AI 插帧
│       ├── ffmpeg_minterp.py     #   FFmpeg 运动补偿插帧
│       └── batch.py              #   批量插帧入口
├── input/                        # 放入待处理的视频
└── output/                       # 处理结果输出
```