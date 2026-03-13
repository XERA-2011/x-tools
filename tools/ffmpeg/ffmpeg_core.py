"""
FFmpeg 拼接/单视频处理的共享构建逻辑
"""
from pathlib import Path

from tools.common import get_video_info


def build_base_filters(
    video_paths: list[Path],
    target_w: int,
    target_h: int,
    trim_start: float,
    trim_end: float,
    audio_fade_in: float,
    audio_fade_out: float,
    include_audio: bool,
) -> tuple[list[str], list[float], list[float]]:
    """
    构建基础滤镜 (scale/pad/trim/afade)

    Returns:
        filter_parts, durations, raw_durations
    """
    filter_parts: list[str] = []
    raw_durations: list[float] = []
    durations: list[float] = []

    for i, vp in enumerate(video_paths):
        info = get_video_info(str(vp))
        raw_dur = info.get("duration", 5.0)
        raw_durations.append(raw_dur)
        durations.append(max(0.1, raw_dur - trim_start - trim_end))

        # 视频流: trim → scale + pad
        v_filters = []
        if trim_start > 0 or trim_end > 0:
            t_end = max(trim_start + 0.1, raw_dur - trim_end)
            v_filters.append(f"trim=start={trim_start}:end={t_end:.3f}")
            v_filters.append("setpts=PTS-STARTPTS")
        v_filters.append(
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease"
        )
        v_filters.append(
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
        )
        v_filters.append("setsar=1")
        filter_parts.append(f"[{i}:v]{','.join(v_filters)}[v{i}]")

        if include_audio:
            # 音频流: atrim + afade
            a_filters = []
            if trim_start > 0 or trim_end > 0:
                t_end = max(trim_start + 0.1, raw_dur - trim_end)
                a_filters.append(f"atrim=start={trim_start}:end={t_end:.3f}")
                a_filters.append("asetpts=PTS-STARTPTS")

            cur_dur = durations[i]
            if cur_dur > audio_fade_in + audio_fade_out:
                if audio_fade_in > 0:
                    a_filters.append(f"afade=t=in:st=0:d={audio_fade_in}:curve=tri")
                if audio_fade_out > 0:
                    a_filters.append(
                        f"afade=t=out:st={cur_dur - audio_fade_out:.3f}:d={audio_fade_out}:curve=tri"
                    )

            if not a_filters:
                a_filters.append("anull")

            filter_parts.append(f"[{i}:a]{','.join(a_filters)}[a{i}]")

    return filter_parts, durations, raw_durations


def build_concat_graph(
    filter_parts: list[str],
    n: int,
    durations: list[float],
    xfade_type: str | None,
    transition_duration: float,
    include_audio: bool,
) -> list[str]:
    """
    构建拼接/过渡的滤镜图，生成 [outv] / [outa]
    """
    td = transition_duration

    if xfade_type and n >= 2:
        cumulative_offset = durations[0] - td
        prev_label = "v0"

        for i in range(1, n):
            out_label = "outv" if i == n - 1 else f"xf{i-1}"
            filter_parts.append(
                f"[{prev_label}][v{i}]xfade=transition={xfade_type}:"
                f"duration={td}:offset={cumulative_offset:.3f}[{out_label}]"
            )
            prev_label = out_label
            if i < n - 1:
                cumulative_offset += durations[i] - td

        if include_audio:
            prev_a = "a0"
            for i in range(1, n):
                out_a = "outa" if i == n - 1 else f"af{i-1}"
                filter_parts.append(
                    f"[{prev_a}][a{i}]acrossfade=d={td}:c1=tri:c2=tri[{out_a}]"
                )
                prev_a = out_a
    elif n == 1:
        filter_parts.append("[v0]null[outv]")
        if include_audio:
            filter_parts.append("[a0]anull[outa]")
    else:
        video_streams = "".join(f"[v{i}]" for i in range(n))
        filter_parts.append(f"{video_streams}concat=n={n}:v=1:a=0[outv]")

        if include_audio:
            audio_streams = "".join(f"[a{i}]" for i in range(n))
            filter_parts.append(f"{audio_streams}concat=n={n}:v=0:a=1[outa]")

    return filter_parts


def compute_total_duration(
    durations: list[float],
    xfade_type: str | None,
    transition_duration: float,
    n: int,
) -> float:
    if xfade_type and n >= 2:
        return sum(durations) - (n - 1) * transition_duration
    return sum(durations)
