"""
音乐节拍检测模块

支持两种后端:
  - librosa (推荐, 精准节拍/onset/BPM 检测)
  - FFmpeg (轻量 fallback, 基于音量包络分析)
"""
import subprocess
import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from config import FFMPEG_BIN, FFPROBE_BIN
from tools.common import logger


@dataclass
class BeatInfo:
    """节拍分析结果"""
    bpm: float = 120.0
    beat_times: list[float] = field(default_factory=list)     # 节拍时间点 (秒)
    onset_times: list[float] = field(default_factory=list)    # 音符起始时间点 (秒)
    volume_envelope: list[float] = field(default_factory=list) # 归一化的音量包络 (0~1)
    duration: float = 0.0                                      # 音频总时长
    backend: str = ""

    def get_beat_intensity(self, time: float, decay: float = 0.15) -> float:
        """
        获取指定时间的节拍强度 (0.0~1.0)

        越接近节拍点, 强度越大。使用指数衰减。

        Args:
            time: 当前时间 (秒)
            decay: 衰减速率 (秒), 值越小衰减越快
        """
        import math
        max_intensity = 0.0
        for bt in self.beat_times:
            dt = time - bt
            if dt < 0:
                break
            if dt < decay * 5:  # 仅计算附近的节拍
                intensity = math.exp(-dt / decay)
                max_intensity = max(max_intensity, intensity)
        return min(max_intensity, 1.0)

    def get_volume_at(self, time: float) -> float:
        """获取指定时间的音量 (0~1)"""
        if not self.volume_envelope or self.duration <= 0:
            return 0.5
        idx = int(time / self.duration * len(self.volume_envelope))
        idx = max(0, min(idx, len(self.volume_envelope) - 1))
        return self.volume_envelope[idx]


def detect_beats(audio_path: str | Path) -> BeatInfo:
    """
    检测音乐节拍 (自动选择最佳后端)

    优先使用 librosa, 不可用时降级到 FFmpeg
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    try:
        return _detect_beats_librosa(audio_path)
    except ImportError:
        logger.warning("librosa 不可用, 降级到 FFmpeg 音量分析")
        return _detect_beats_ffmpeg(audio_path)


def _detect_beats_librosa(audio_path: Path) -> BeatInfo:
    """使用 librosa 进行精准节拍检测"""
    import librosa
    import numpy as np

    logger.info("使用 librosa 分析节拍...")

    # 加载音频
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # BPM 和节拍检测
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # librosa 0.10+ tempo 可能是数组
    if hasattr(tempo, '__len__'):
        bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        bpm = float(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # Onset 检测 (音符起始)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    # RMS 音量包络 (归一化)
    rms = librosa.feature.rms(y=y)[0]
    rms_max = rms.max() if rms.max() > 0 else 1.0
    volume_envelope = (rms / rms_max).tolist()

    logger.info(f"librosa 分析完成: BPM={bpm:.0f}, {len(beat_times)} 个节拍, "
                f"{len(onset_times)} 个 onset")

    return BeatInfo(
        bpm=bpm,
        beat_times=beat_times,
        onset_times=onset_times,
        volume_envelope=volume_envelope,
        duration=duration,
        backend="librosa",
    )


def _detect_beats_ffmpeg(audio_path: Path) -> BeatInfo:
    """
    使用 FFmpeg 进行音量包络分析 (fallback)

    通过 astats 滤镜提取 RMS 音量, 再用峰值检测近似节拍
    """
    logger.info("使用 FFmpeg 分析音量包络...")

    # 获取音频时长
    duration = _get_audio_duration(audio_path)

    # 使用 FFmpeg volumedetect 获取整体信息
    # 使用 astats 逐帧分析获取音量包络
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(audio_path),
        "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
        "-f", "null", "-",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )

    # 解析 RMS 值
    rms_values = []
    rms_pattern = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(-?\d+\.?\d*)")

    output = result.stdout + result.stderr
    for m in rms_pattern.finditer(output):
        val = float(m.group(1))
        # dB 转线性, 归一化
        if val > -100:
            import math
            linear = math.pow(10, val / 20)
            rms_values.append(linear)
        else:
            rms_values.append(0.0)

    # 归一化音量包络
    if rms_values:
        max_val = max(rms_values) if max(rms_values) > 0 else 1.0
        volume_envelope = [v / max_val for v in rms_values]
    else:
        # FFmpeg 分析失败, 生成模拟包络
        logger.warning("FFmpeg 音量分析无输出, 使用默认 120 BPM 模拟节拍")
        volume_envelope = [0.5] * 100

    # 从音量包络中检测节拍 (简单峰值检测)
    beat_times = _detect_peaks_from_envelope(volume_envelope, duration)

    # 估算 BPM
    if len(beat_times) >= 2:
        intervals = [beat_times[i + 1] - beat_times[i] for i in range(len(beat_times) - 1)]
        avg_interval = sum(intervals) / len(intervals)
        bpm = 60.0 / avg_interval if avg_interval > 0 else 120.0
    else:
        bpm = 120.0

    logger.info(f"FFmpeg 分析完成: 估计 BPM={bpm:.0f}, {len(beat_times)} 个节拍")

    return BeatInfo(
        bpm=bpm,
        beat_times=beat_times,
        onset_times=beat_times,  # FFmpeg 模式下 onset 等同于 beat
        volume_envelope=volume_envelope,
        duration=duration,
        backend="ffmpeg",
    )


def _detect_peaks_from_envelope(envelope: list[float], duration: float) -> list[float]:
    """
    从音量包络中检测峰值点作为节拍

    使用简单的局部极大值检测
    """
    if len(envelope) < 3 or duration <= 0:
        return []

    time_per_sample = duration / len(envelope)

    # 计算自适应阈值 (中位数 + 0.3 * 标准差)
    sorted_env = sorted(envelope)
    median = sorted_env[len(sorted_env) // 2]
    mean = sum(envelope) / len(envelope)
    std = (sum((v - mean) ** 2 for v in envelope) / len(envelope)) ** 0.5
    threshold = median + 0.3 * std

    # 最小间距 (相当于最快 200 BPM)
    min_gap_samples = max(1, int(0.3 / time_per_sample))

    peaks = []
    last_peak_idx = -min_gap_samples

    for i in range(1, len(envelope) - 1):
        if (
            envelope[i] > envelope[i - 1]
            and envelope[i] >= envelope[i + 1]
            and envelope[i] > threshold
            and i - last_peak_idx >= min_gap_samples
        ):
            peaks.append(i * time_per_sample)
            last_peak_idx = i

    return peaks


def _get_audio_duration(audio_path: Path) -> float:
    """获取音频时长"""
    cmd = [
        FFPROBE_BIN, "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(audio_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", check=True,
        )
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        logger.warning("无法获取音频时长, 默认 30 秒")
        return 30.0
