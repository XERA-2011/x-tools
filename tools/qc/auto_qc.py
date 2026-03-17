"""
自动质量检测 (QC)

功能:
  - 读取视频元信息 (分辨率/帧率/码率/音频等)
  - 黑场检测 (blackdetect)
  - 静音检测 (silencedetect)
  - 冻结检测 (freezedetect)
  - 生成可读报告 + 保存 JSON
"""
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from config import (
    FFMPEG_BIN,
    OUTPUT_QC,
    QC_BLACK_SEGMENT_WARN_SEC,
    QC_BLACK_TOTAL_WARN_SEC,
    QC_FREEZE_SEGMENT_WARN_SEC,
    QC_FREEZE_TOTAL_WARN_SEC,
    QC_MIN_AUDIO_SAMPLE_RATE,
    QC_MIN_BITRATE_1080P,
    QC_MIN_BITRATE_4K,
    QC_MIN_BITRATE_720P,
    QC_MIN_FPS,
    QC_MIN_HEIGHT,
    QC_MIN_WIDTH,
    QC_SILENCE_SEGMENT_WARN_SEC,
    QC_SILENCE_TOTAL_WARN_SEC,
)
from tools.common import batch_process, generate_output_name, logger
from tools.mediainfo.probe import format_bitrate, format_duration, format_filesize, get_detailed_info

console = Console()

BLACK_RE = re.compile(r"black_start:(?P<start>[\d\.]+)\s+black_end:(?P<end>[\d\.]+)\s+black_duration:(?P<dur>[\d\.]+)")
SILENCE_START_RE = re.compile(r"silence_start:\s*(?P<start>[\d\.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*(?P<end>[\d\.]+)\s*\|\s*silence_duration:\s*(?P<dur>[\d\.]+)")
FREEZE_START_RE = re.compile(r"freeze_start:\s*(?P<start>[\d\.]+)")
FREEZE_END_RE = re.compile(r"freeze_end:\s*(?P<end>[\d\.]+)\s*\|\s*freeze_duration:\s*(?P<dur>[\d\.]+)")


def _run_ffmpeg_detect(video_path: Path, vf: str | None = None, af: str | None = None) -> str:
    cmd = [FFMPEG_BIN, "-hide_banner", "-v", "info", "-i", str(video_path)]
    if vf:
        cmd += ["-vf", vf]
    if af:
        cmd += ["-af", af]
    cmd += ["-f", "null", "-"]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg 执行失败")
    return result.stderr


def _parse_black(stderr: str) -> list[dict]:
    segments = []
    for line in stderr.splitlines():
        m = BLACK_RE.search(line)
        if m:
            segments.append({
                "start": float(m.group("start")),
                "end": float(m.group("end")),
                "duration": float(m.group("dur")),
            })
    return segments


def _parse_silence(stderr: str, duration: float) -> list[dict]:
    segments = []
    current_start: float | None = None
    for line in stderr.splitlines():
        m_start = SILENCE_START_RE.search(line)
        if m_start:
            current_start = float(m_start.group("start"))
            continue
        m_end = SILENCE_END_RE.search(line)
        if m_end:
            start = current_start if current_start is not None else float(m_end.group("end"))
            end = float(m_end.group("end"))
            segments.append({"start": start, "end": end, "duration": float(m_end.group("dur"))})
            current_start = None
    if current_start is not None and duration > 0:
        segments.append({"start": current_start, "end": duration, "duration": max(0.0, duration - current_start)})
    return segments


def _parse_freeze(stderr: str, duration: float) -> list[dict]:
    segments = []
    current_start: float | None = None
    for line in stderr.splitlines():
        m_start = FREEZE_START_RE.search(line)
        if m_start:
            current_start = float(m_start.group("start"))
            continue
        m_end = FREEZE_END_RE.search(line)
        if m_end:
            start = current_start if current_start is not None else float(m_end.group("end"))
            end = float(m_end.group("end"))
            segments.append({"start": start, "end": end, "duration": float(m_end.group("dur"))})
            current_start = None
    if current_start is not None and duration > 0:
        segments.append({"start": current_start, "end": duration, "duration": max(0.0, duration - current_start)})
    return segments


def _sum_duration(segments: list[dict]) -> float:
    return round(sum(s.get("duration", 0.0) for s in segments), 2)


def _bitrate_warn_threshold(width: int, height: int) -> int:
    pixels = max(width, height)
    if pixels >= 2160:
        return QC_MIN_BITRATE_4K
    if pixels >= 1080:
        return QC_MIN_BITRATE_1080P
    return QC_MIN_BITRATE_720P


def _collect_warnings(info: dict, black: list[dict], silence: list[dict], freeze: list[dict]) -> list[str]:
    warnings: list[str] = []
    video = info.get("video") or {}
    audio = info.get("audio")

    width = int(video.get("width", 0))
    height = int(video.get("height", 0))
    fps = float(video.get("fps", 0))
    bitrate = int(info.get("bitrate", 0))

    if width and height and (width < QC_MIN_WIDTH or height < QC_MIN_HEIGHT):
        warnings.append("分辨率低于 720p")
    if fps and fps < QC_MIN_FPS:
        warnings.append("帧率偏低")
    if bitrate and width and height:
        threshold = _bitrate_warn_threshold(width, height)
        if bitrate < threshold:
            warnings.append("码率偏低")

    if not audio:
        warnings.append("无音频流")
    else:
        sample_rate = int(audio.get("sample_rate", 0))
        if sample_rate and sample_rate < QC_MIN_AUDIO_SAMPLE_RATE:
            warnings.append("音频采样率偏低")

    black_total = _sum_duration(black)
    silence_total = _sum_duration(silence)
    freeze_total = _sum_duration(freeze)

    if black_total >= QC_BLACK_TOTAL_WARN_SEC:
        warnings.append("黑场较多")
    if any(s["duration"] >= QC_BLACK_SEGMENT_WARN_SEC for s in black):
        warnings.append("存在长黑场")

    if silence_total >= QC_SILENCE_TOTAL_WARN_SEC:
        warnings.append("静音较多")
    if any(s["duration"] >= QC_SILENCE_SEGMENT_WARN_SEC for s in silence):
        warnings.append("存在长静音")

    if freeze_total >= QC_FREEZE_TOTAL_WARN_SEC:
        warnings.append("冻结较多")
    if any(s["duration"] >= QC_FREEZE_SEGMENT_WARN_SEC for s in freeze):
        warnings.append("存在长冻结")

    return warnings


def analyze_video_qc(
    video_path: Path,
    detect_black: bool = True,
    detect_silence: bool = True,
    detect_freeze: bool = True,
) -> dict[str, Any]:
    info = get_detailed_info(video_path)
    if not info:
        raise RuntimeError("无法读取媒体信息")

    duration = float(info.get("duration", 0))
    has_video = info.get("video") is not None
    has_audio = info.get("audio") is not None
    extra_warnings: list[str] = []

    black_segments: list[dict] = []
    silence_segments: list[dict] = []
    freeze_segments: list[dict] = []

    if has_video and (detect_black or detect_freeze):
        vf_parts = []
        if detect_black:
            vf_parts.append("blackdetect=d=0.5:pic_th=0.98")
        if detect_freeze:
            vf_parts.append("freezedetect=n=0.003:d=2")
        vf = ",".join(vf_parts) if vf_parts else None
        try:
            stderr = _run_ffmpeg_detect(video_path, vf=vf)
            if detect_black:
                black_segments = _parse_black(stderr)
            if detect_freeze:
                freeze_segments = _parse_freeze(stderr, duration)
        except Exception as e:
            logger.warning(f"黑场/冻结检测失败: {video_path.name} — {e}")
            extra_warnings.append("黑场/冻结检测失败")

    if has_audio and detect_silence:
        try:
            stderr = _run_ffmpeg_detect(video_path, af="silencedetect=n=-50dB:d=1")
            silence_segments = _parse_silence(stderr, duration)
        except Exception as e:
            logger.warning(f"静音检测失败: {video_path.name} — {e}")
            extra_warnings.append("静音检测失败")

    warnings = _collect_warnings(info, black_segments, silence_segments, freeze_segments)
    warnings.extend(extra_warnings)

    report = {
        "file": str(video_path),
        "filename": video_path.name,
        "filesize": info.get("filesize", 0),
        "container": info.get("container", ""),
        "duration": duration,
        "bitrate": info.get("bitrate", 0),
        "video": info.get("video"),
        "audio": info.get("audio"),
        "black_segments": black_segments,
        "silence_segments": silence_segments,
        "freeze_segments": freeze_segments,
        "black_total": _sum_duration(black_segments),
        "silence_total": _sum_duration(silence_segments),
        "freeze_total": _sum_duration(freeze_segments),
        "warnings": warnings,
    }
    return report


def _save_report(report: dict) -> Path:
    OUTPUT_QC.mkdir(parents=True, exist_ok=True)
    stem = Path(report["filename"]).stem
    out_name = generate_output_name(stem, ".json", tag="qc")
    out_path = OUTPUT_QC / out_name
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _format_seconds(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    return f"{seconds:.1f}s"


def display_qc_summary(results: list[dict]) -> None:
    if not results:
        return

    table = Table(
        title="✅ 自动质量检测汇总",
        border_style="cyan",
        title_style="bold cyan",
    )
    table.add_column("文件名", style="bold", max_width=28)
    table.add_column("分辨率", width=12)
    table.add_column("帧率", width=8)
    table.add_column("码率", width=10)
    table.add_column("音频", width=8)
    table.add_column("黑场", width=8)
    table.add_column("静音", width=8)
    table.add_column("冻结", width=8)
    table.add_column("问题", width=6)

    for r in results:
        if r["status"] != "success":
            table.add_row(Path(r["file"]).name, "—", "—", "—", "—", "—", "—", "—", "❌")
            continue

        report = r.get("report", {})
        video = report.get("video") or {}
        audio = report.get("audio")

        res = f"{video.get('width', '?')}×{video.get('height', '?')}" if video else "—"
        fps = f"{video.get('fps', 0)}" if video else "—"
        bitrate = format_bitrate(int(report.get("bitrate", 0))) if report.get("bitrate") else "未知"
        audio_str = audio.get("codec", "无").upper() if audio else "无"

        table.add_row(
            Path(r["file"]).name,
            res,
            fps,
            bitrate,
            audio_str,
            _format_seconds(report.get("black_total", 0.0)),
            _format_seconds(report.get("silence_total", 0.0)),
            _format_seconds(report.get("freeze_total", 0.0)),
            str(len(report.get("warnings", []))),
        )

    console.print()
    console.print(table)
    console.print()

    if len(results) == 1 and results[0]["status"] == "success":
        _display_qc_detail(results[0]["report"])


def _display_qc_detail(report: dict) -> None:
    table = Table(
        title=f"📋 质量检测详情: {report['filename']}",
        border_style="cyan",
        title_style="bold cyan",
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("属性", style="bold", width=16)
    table.add_column("值", min_width=30)

    video = report.get("video") or {}
    audio = report.get("audio") or {}

    table.add_row("📁 文件大小", format_filesize(report.get("filesize", 0)))
    table.add_row("📦 容器格式", report.get("container", ""))
    table.add_row("⏱️  时长", format_duration(report.get("duration", 0)))
    table.add_row("📊 总码率", format_bitrate(int(report.get("bitrate", 0))) if report.get("bitrate") else "未知")

    if video:
        res = f"{video.get('width', '?')}×{video.get('height', '?')}"
        table.add_row("🎬 分辨率", res)
        table.add_row("🎞️  帧率", f"{video.get('fps', 0)} FPS")
        table.add_row("🔧 视频编码", str(video.get("codec", "")).upper())

    if audio:
        table.add_row("🔊 音频编码", str(audio.get("codec", "")).upper())
        if audio.get("sample_rate"):
            table.add_row("🎵 采样率", f"{audio.get('sample_rate'):,} Hz")
        if audio.get("channels"):
            table.add_row("📢 声道", f"{audio.get('channels')} 声道")
    else:
        table.add_row("🔇 音频", "无音频流")

    table.add_section()
    table.add_row("🖤 黑场总时长", _format_seconds(report.get("black_total", 0.0)))
    table.add_row("🔇 静音总时长", _format_seconds(report.get("silence_total", 0.0)))
    table.add_row("🧊 冻结总时长", _format_seconds(report.get("freeze_total", 0.0)))

    console.print(table)

    warnings = report.get("warnings", [])
    if warnings:
        console.print("\n⚠️ 发现问题:")
        for w in warnings:
            console.print(f"  - {w}")
    else:
        console.print("\n✅ 未发现明显问题")

    console.print()


def _qc_worker(video_path: Path, detect_black: bool, detect_silence: bool, detect_freeze: bool) -> dict:
    report = analyze_video_qc(
        video_path,
        detect_black=detect_black,
        detect_silence=detect_silence,
        detect_freeze=detect_freeze,
    )
    report_path = _save_report(report)
    return {"report": report, "report_path": str(report_path)}


def batch_qc(
    videos: list[Path],
    detect_black: bool = True,
    detect_silence: bool = True,
    detect_freeze: bool = True,
) -> list[dict]:
    results = batch_process(
        videos,
        _qc_worker,
        desc="自动质量检测 (QC)",
        max_workers=1,
        detect_black=detect_black,
        detect_silence=detect_silence,
        detect_freeze=detect_freeze,
    )

    success = sum(1 for r in results if r["status"] == "success")
    logger.info(f"QC 完成: ✅ {success}/{len(results)} 成功")
    return results


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="自动质量检测 (QC)")
    parser.add_argument("files", nargs="+", help="输入视频文件路径")
    parser.add_argument("--no-black", action="store_true", help="禁用黑场检测")
    parser.add_argument("--no-silence", action="store_true", help="禁用静音检测")
    parser.add_argument("--freeze", action="store_true", help="启用冻结检测 (更慢)")

    args = parser.parse_args()
    files = [Path(f) for f in args.files]

    results = batch_qc(
        files,
        detect_black=not args.no_black,
        detect_silence=not args.no_silence,
        detect_freeze=args.freeze,
    )
    display_qc_summary(results)
