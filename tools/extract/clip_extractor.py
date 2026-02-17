"""
视频片段截取模块

功能:
  - 按时间范围截取视频片段
  - 支持多段截取
  - 保留原始画质 (stream copy) 或重新编码
"""
import subprocess
from pathlib import Path


from config import FFMPEG_BIN, OUTPUT_EXTRACT, EXTRACT_DEFAULT_FORMAT
from tools.common import get_video_info, logger


def extract_clip(
    video_path: str | Path,
    start: str,
    end: str | None = None,
    duration: str | None = None,
    output_path: str | Path | None = None,
    reencode: bool = False,
) -> dict:
    """
    截取视频片段

    Args:
        video_path: 输入视频路径
        start: 开始时间 (格式: "HH:MM:SS" 或 "SS" 或 "HH:MM:SS.mmm")
        end: 结束时间 (与 duration 二选一)
        duration: 持续时长 (与 end 二选一)
        output_path: 输出路径, 为 None 时自动生成
        reencode: 是否重新编码 (False=stream copy, 更快但可能不精确)

    Returns:
        dict: {"output": str, "duration": float}
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if end is None and duration is None:
        raise ValueError("必须指定 end 或 duration 其中之一")

    # 生成输出路径
    if output_path is None:
        OUTPUT_EXTRACT.mkdir(parents=True, exist_ok=True)
        suffix = f".{EXTRACT_DEFAULT_FORMAT}"
        stem = video_path.stem
        time_tag = start.replace(":", "").replace(".", "")
        output_path = OUTPUT_EXTRACT / f"{stem}_clip_{time_tag}{suffix}"
    output_path = Path(output_path)

    # 构建 FFmpeg 命令
    cmd = [FFMPEG_BIN, "-y"]

    # 输入定位 (放在 -i 前面以利用 fast seek)
    cmd += ["-ss", start]
    cmd += ["-i", str(video_path)]

    # 时长/结束时间
    if duration:
        cmd += ["-t", duration]
    elif end:
        cmd += ["-to", end]

    # 编码模式
    if reencode:
        cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "fast"]
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-c", "copy"]

    # 避免负时间戳问题
    cmd += ["-avoid_negative_ts", "make_zero"]

    cmd += [str(output_path)]

    logger.info(f"截取片段: {video_path.name} [{start} → {end or f'+{duration}'}]")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 错误:\n{result.stderr[-500:]}")

    # 获取输出文件信息
    info = get_video_info(output_path)

    logger.info(f"✅ 输出: {output_path.name} ({info.get('duration', 0):.1f}s)")

    return {"output": str(output_path), "duration": info.get("duration", 0)}


def extract_multi_clips(
    video_path: str | Path,
    segments: list[dict],
    reencode: bool = False,
) -> list[dict]:
    """
    从同一视频截取多个片段

    Args:
        video_path: 输入视频路径
        segments: 片段列表, 每项为 {"start": str, "end": str} 或 {"start": str, "duration": str}
        reencode: 是否重新编码

    Returns:
        list[dict]: 每个片段的处理结果
    """
    results = []
    for i, seg in enumerate(segments, 1):
        logger.info(f"正在截取第 {i}/{len(segments)} 个片段...")
        try:
            result = extract_clip(
                video_path,
                start=seg["start"],
                end=seg.get("end"),
                duration=seg.get("duration"),
                reencode=reencode,
            )
            results.append({"segment": i, "status": "success", **result})
        except Exception as e:
            logger.error(f"片段 {i} 截取失败: {e}")
            results.append({"segment": i, "status": "error", "error": str(e)})

    return results


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="视频片段截取")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("-s", "--start", required=True, help="开始时间 (HH:MM:SS)")
    parser.add_argument("-e", "--end", help="结束时间 (HH:MM:SS)")
    parser.add_argument("-d", "--duration", help="持续时长 (HH:MM:SS 或秒数)")
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument("--reencode", action="store_true", help="重新编码 (更精确但更慢)")

    args = parser.parse_args()

    result = extract_clip(
        video_path=args.input,
        start=args.start,
        end=args.end,
        duration=args.duration,
        output_path=args.output,
        reencode=args.reencode,
    )
    print(f"\n✅ 截取完成: {result['output']}")
