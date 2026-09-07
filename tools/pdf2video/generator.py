"""
PDF 转视频核心合成引擎 (PDF to Video Generator)

功能:
  - 串联 PDF 正文提取、页面高清渲染、Edge-TTS 异步并发配音
  - 动态音画时长对齐 (每页时长严格由语音时长 + 缓冲留白控制)
  - 逐页微切片渲染 + FFmpeg 毫秒级 Concat 快速合并
  - 支持同步导出对齐的 .srt 字幕文件 (默认画面不硬烧录，避免遮挡)
  - 支持轻柔 BGM 混音与淡出
"""
import asyncio
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from config import FFMPEG_BIN, OUTPUT_PDF2VIDEO
from tools.common import console, generate_output_name, get_video_info, logger
from tools.pdf2video.extractor import extract_script_from_pdf
from tools.pdf2video.renderer import render_pdf_to_images
from tools.slideshow.generator import resize_and_pad
from tools.tts.tts_generate import TTS_VOICES


def _format_srt_time(seconds: float) -> str:
    """将秒数转为 SRT 时间戳 00:00:00,000"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis = 0
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _format_ass_time(sec: float) -> str:
    """将秒数转为 ASS 时间戳 0:00:00.00"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs >= 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _wrap_text_ass(text: str, max_chars_per_line: int = 36, max_lines: int = 2) -> str:
    """
    按中文标点与字符宽度智能均衡折行，避免孤立标点和排版割裂
    """
    single_line = " ".join(text.split())
    if not single_line:
        return ""

    def get_width(s: str) -> int:
        return sum(2 if ord(c) > 255 else 1 for c in s)

    total_w = get_width(single_line)
    max_w = max_chars_per_line * 2

    # 单行放得下直接返回单行
    if total_w <= max_w or max_lines <= 1:
        return single_line

    # 尝试在文本中间寻找最合适的标点断句点，使两行长度最为均衡
    puncts = "，。！？；、 "
    mid = len(single_line) // 2
    best_split = -1
    best_diff = float("inf")

    search_start = max(1, mid - 14)
    search_end = min(len(single_line) - 1, mid + 14)
    for idx in range(search_start, search_end):
        if single_line[idx] in puncts:
            w1 = get_width(single_line[: idx + 1])
            w2 = get_width(single_line[idx + 1 :])
            if w1 <= max_w and w2 <= max_w:
                diff = abs(w1 - w2)
                if diff < best_diff:
                    best_diff = diff
                    best_split = idx + 1

    if best_split > 0:
        line1 = single_line[:best_split].strip()
        line2 = single_line[best_split:].strip()
        return line1 + r"\N" + line2

    # 无标点时居中对称折行，避开行首标点
    avoid_start = "，。！？；、）》」』”’"
    split_idx = mid
    while split_idx < len(single_line) - 1 and single_line[split_idx] in avoid_start:
        split_idx += 1

    line1 = single_line[:split_idx].strip()
    line2 = single_line[split_idx:].strip()
    return line1 + r"\N" + line2


def build_single_slide_ass(text: str, duration: float, resolution: tuple[int, int], output_path: Path) -> Path:
    """
    为单页生成底部精致小字幕 ASS 文件 (最多 2 行，贴底显示，避免遮挡页面)
    """
    width, height = resolution
    font_size = max(18, int(height // 40))
    margin_v = int(height * 0.03)
    margin_h = int(width * 0.08)

    format_line = (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
    )
    # 使用半透明微黑底框 (BorderStyle=3, Outline=6 作为内边距, BackColour=&H70000000), 贴底居中, 绝不遮挡主体
    style_line = (
        f"Style: Default,PingFang SC,{font_size},&H00FFFFFF,&H000000FF,&H00000000,"
        f"&H70000000,-1,0,0,0,100,100,0,0,3,6,0,2,{margin_h},{margin_h},{margin_v},1"
    )

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 1

[V4+ Styles]
{format_line}
{style_line}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    available_width = width - (margin_h * 2)
    max_chars = max(12, int(available_width / font_size))
    formatted_text = _wrap_text_ass(text, max_chars_per_line=max_chars, max_lines=2)

    start_str = _format_ass_time(0.0)
    end_str = _format_ass_time(duration)
    event_line = f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{formatted_text}"

    ass_content = ass_header + event_line + "\n"
    output_path.write_text(ass_content, encoding="utf-8")
    return output_path


async def _generate_slide_tts(
    text: str,
    voice_id: str,
    output_path: Path,
    sem: asyncio.Semaphore | None = None,
    max_retries: int = 3,
    on_complete: Callable | None = None,
):
    """异步调用 edge-tts 生成单页音频 (支持信号量并发限制与重试机制)"""
    import edge_tts
    for attempt in range(max_retries):
        try:
            if sem:
                async with sem:
                    communicate = edge_tts.Communicate(text, voice_id)
                    await communicate.save(str(output_path))
            else:
                communicate = edge_tts.Communicate(text, voice_id)
                await communicate.save(str(output_path))
            if on_complete:
                on_complete()
            return
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"TTS 生成失败 (尝试 {max_retries} 次): {e}")
                raise e
            await asyncio.sleep(0.4 * (attempt + 1))


def generate_pdf_video(
    pdf_path: Path | str,
    voice_key: str | None = None,
    voice: str | None = None,
    padding_after: float | None = None,
    page_padding: float | None = None,
    burn_subtitles: bool = True,
    resolution: tuple[int, int] | str = (1920, 1080),
    music_path: Path | str | None = None,
    bgm_path: Path | str | None = None,
    music_volume: float | None = None,
    bgm_volume: float | None = None,
    output_path: Path | str | None = None,
    custom_script: list[dict] | None = None,
    show_progress: bool = True,
) -> dict:
    """
    将 PDF 转换为每一页均有人声朗读的高清视频

    Args:
        pdf_path: PDF 文件路径
        voice_key: 配音音色 key (见 TTS_VOICES) 或完整 voice 名称
        voice: voice_key 别名
        padding_after: 朗读完毕后留在当前页的静音留白秒数 (默认 0.8s)
        page_padding: padding_after 别名
        burn_subtitles: 是否将朗读字幕烧录在视频画面上 (默认 True, 使用半透明贴底质感字幕框)
        resolution: 最终视频分辨率 (宽, 高) 或 "1080p" / "2k" / "720p", 默认 (1920, 1080)
        music_path: 背景音乐文件路径
        bgm_path: music_path 别名
        music_volume: 背景音乐音量 (0.0~1.0, 默认 0.15)
        bgm_volume: music_volume 别名
        output_path: 指定输出路径 (若无则自动在 output/pdf2video/ 下生成)
        custom_script: 外置预加载的文案列表
        show_progress: 是否显示终端丰富进度条 (默认 True)

    Returns:
        {
            "output": 视频文件路径,
            "srt": 字幕文件路径,
            "duration": 总时长秒数,
            "slides_count": 总页数,
            "size_mb": 视频大小 MB
        }
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"未找到 PDF 文件: {pdf_path}")

    # 0. 参数归一化
    v_param = voice or voice_key or "xiaoxiao"
    if v_param in TTS_VOICES:
        voice_id = TTS_VOICES[v_param]["voice"]
        voice_label = TTS_VOICES[v_param].get("name", voice_id)
    elif any(v.get("voice") == v_param for v in TTS_VOICES.values()):
        voice_id = v_param
        voice_label = v_param
    elif v_param.startswith("zh-") or v_param.startswith("en-"):
        voice_id = v_param
        voice_label = v_param
    else:
        voice_id = "zh-CN-YunxiNeural"
        voice_label = "云希 (男声)"

    padding_val = page_padding if page_padding is not None else (padding_after if padding_after is not None else 0.8)
    padding_after = padding_val

    chosen_music_path = bgm_path or music_path
    chosen_music_volume = bgm_volume if bgm_volume is not None else (music_volume if music_volume is not None else 0.15)

    if isinstance(resolution, str):
        res_str = resolution.lower()
        if res_str in ("1080p", "1080"):
            resolution = (1920, 1080)
        elif res_str in ("2k", "1440p"):
            resolution = (2560, 1440)
        elif res_str in ("720p", "720"):
            resolution = (1280, 720)
        else:
            resolution = (1920, 1080)

    # 1. 准备输出路径
    OUTPUT_PDF2VIDEO.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        out_name = generate_output_name(f"pdf_{pdf_path.stem}", ".mp4")
        output_path = OUTPUT_PDF2VIDEO / out_name
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    srt_output_path = output_path.with_suffix(".srt")

    # 3. 创建临时工作空间
    temp_dir = Path(tempfile.mkdtemp(prefix="x-tools_pdf2video_"))
    slides_img_dir = temp_dir / "images"
    audio_dir = temp_dir / "audios"
    segments_dir = temp_dir / "segments"
    slides_img_dir.mkdir()
    audio_dir.mkdir()
    segments_dir.mkdir()

    try:
        # ============================================================
        # 步骤 A: 提取文案
        # ============================================================
        if custom_script:
            script = custom_script
            logger.info(f"使用用户自定义文案 (共 {len(script)} 页)")
        else:
            script = extract_script_from_pdf(pdf_path)

        # 预估总页数以初始化进度条
        total_est = len(script) if script else None
        if total_est is None:
            try:
                import pypdfium2 as pdfium
                _doc = pdfium.PdfDocument(str(pdf_path))
                total_est = len(_doc)
                _doc.close()
            except Exception:
                total_est = None

        progress_cm = (
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=console,
            )
            if show_progress
            else nullcontext()
        )

        with progress_cm as progress:
            task_id = (
                progress.add_task("[cyan][1/4] 📄 渲染 PDF 页面[/cyan]", total=total_est)
                if progress
                else None
            )

            # ============================================================
            # 步骤 B: 渲染 PDF 高清画面
            # ============================================================
            def _on_page_rendered(curr, total):
                if progress and task_id is not None:
                    if total:
                        progress.update(task_id, total=total)
                    progress.advance(task_id)

            image_paths = render_pdf_to_images(
                pdf_path, slides_img_dir, scale=2.0, progress_callback=_on_page_rendered
            )
            total_slides = len(image_paths)
            if total_slides == 0:
                raise RuntimeError("未渲染出任何页面")

            if progress and task_id is not None:
                progress.update(task_id, completed=total_slides, total=total_slides)

            # 对齐文案与页面数
            while len(script) < total_slides:
                script.append({"page": len(script) + 1, "text": "", "source": "empty"})

            # ============================================================
            # 步骤 C: Edge-TTS 并发配音生成
            # ============================================================
            if progress and task_id is not None:
                progress.reset(
                    task_id,
                    total=total_slides,
                    description=f"[yellow][2/4] 🎙️ 生成语音配音 ({voice_label})[/yellow]",
                )
            logger.debug(f"正在生成 AI 语音配音 (音色: {voice_label})...")
            tts_tasks = []
            slide_audios: list[dict] = []

            for i in range(total_slides):
                page_num = i + 1
                text = (script[i].get("text") or "").strip()
                audio_file = audio_dir / f"slide_{page_num:04d}.mp3"

                slide_audios.append({
                    "page": page_num,
                    "text": text,
                    "audio_path": audio_file,
                    "duration": 0.0,
                })

                if text:
                    tts_tasks.append((text, audio_file))

            empty_count = total_slides - len(tts_tasks)
            if empty_count > 0 and progress and task_id is not None:
                progress.advance(task_id, advance=empty_count)

            async def _run_all(tasks_data):
                sem = asyncio.Semaphore(4)
                coros = [
                    _generate_slide_tts(
                        t,
                        voice_id,
                        p,
                        sem=sem,
                        on_complete=lambda: progress.advance(task_id) if progress and task_id is not None else None,
                    )
                    for t, p in tasks_data
                ]
                await asyncio.gather(*coros)

            if tts_tasks:
                asyncio.run(_run_all(tts_tasks))

            if progress and task_id is not None:
                progress.update(task_id, completed=total_slides)

            # 测量每个音频文件的时长
            for item in slide_audios:
                p = item["audio_path"]
                if p.is_file() and p.stat().st_size > 0:
                    info = get_video_info(p)
                    dur = float(info.get("duration", 0.0))
                    item["duration"] = dur
                else:
                    item["duration"] = 0.0

            # ============================================================
            # 步骤 D: 计算每页时长 & 生成同步 SRT 字幕文件
            # ============================================================
            default_silent_dur = 3.0
            slide_durations: list[float] = []
            srt_lines: list[str] = []
            current_time = 0.0

            for i, item in enumerate(slide_audios):
                audio_dur = item["duration"]
                text = item["text"]

                if audio_dur > 0:
                    slide_dur = audio_dur + padding_after
                    text_end_time = current_time + audio_dur
                else:
                    slide_dur = default_silent_dur
                    text_end_time = current_time + slide_dur

                slide_durations.append(slide_dur)

                # 生成 SRT 条目 (若该页有文字)
                if text:
                    srt_start = _format_srt_time(current_time)
                    srt_end = _format_srt_time(text_end_time)
                    clean_text = " ".join(text.split())
                    srt_idx = len(srt_lines) + 1
                    srt_lines.append(f"{srt_idx}\n{srt_start} --> {srt_end}\n{clean_text}\n")

                current_time += slide_dur

            total_video_duration = current_time
            if srt_lines:
                srt_output_path.write_text("\n".join(srt_lines), encoding="utf-8")
                logger.debug(f"✅ 同步 SRT 字幕已输出: {srt_output_path.name}")

            # ============================================================
            # 步骤 E: 逐页渲染独立 MP4 切片
            # ============================================================
            if progress and task_id is not None:
                progress.reset(
                    task_id,
                    total=total_slides,
                    description="[green][3/4] 🎬 合成视频切片与字幕[/green]",
                )
            logger.debug(f"开始合成单页视频切片 (共 {total_slides} 页)...")
            target_w, target_h = resolution
            segment_files: list[Path] = []

            for i in range(total_slides):
                page_num = i + 1
                img_p = image_paths[i]
                slide_dur = slide_durations[i]
                audio_dur = slide_audios[i]["duration"]
                audio_file = slide_audios[i]["audio_path"]
                text = slide_audios[i]["text"]

                # 1. 预处理图片到统一分辨率 (保持宽高比并填充黑底)
                fitted_img_path = temp_dir / f"fit_{page_num:04d}.jpg"
                img = Image.open(img_p).convert("RGB")
                fitted_img, _, _, _, _ = resize_and_pad(img, target_w, target_h)
                fitted_img.save(fitted_img_path, "JPEG", quality=95)

                # 2. 单页 ASS 字幕 (如开启)
                ass_path = None
                if burn_subtitles and text:
                    ass_path = temp_dir / f"sub_{page_num:04d}.ass"
                    build_single_slide_ass(text, slide_dur, resolution, ass_path)

                # 3. 构造单页 FFmpeg 命令
                seg_out = segments_dir / f"seg_{page_num:04d}.mp4"
                cmd = [FFMPEG_BIN, "-y"]

                # 输入 0: 静态图片循环
                cmd.extend(["-loop", "1", "-t", f"{slide_dur:.3f}", "-i", str(fitted_img_path)])

                # 输入 1: 音频输入
                if audio_dur > 0 and audio_file.is_file():
                    cmd.extend(["-i", str(audio_file)])
                    filter_complex_parts = []
                    if ass_path:
                        escaped_ass = ass_path.resolve().as_posix().replace(":", "\\:").replace("'", "\\'")
                        filter_complex_parts.append(f"[0:v]subtitles='{escaped_ass}'[v_out]")
                    else:
                        filter_complex_parts.append("[0:v]null[v_out]")

                    filter_complex_parts.append(f"[1:a]apad=pad_dur={padding_after:.3f}[a_out]")

                    cmd.extend([
                        "-filter_complex", ";".join(filter_complex_parts),
                        "-map", "[v_out]",
                        "-map", "[a_out]",
                    ])
                else:
                    # 纯静音占位音频
                    cmd.extend(["-f", "lavfi", "-t", f"{slide_dur:.3f}", "-i", "anullsrc=r=44100:cl=stereo"])
                    if ass_path:
                        escaped_ass = ass_path.resolve().as_posix().replace(":", "\\:").replace("'", "\\'")
                        cmd.extend([
                            "-filter_complex", f"[0:v]subtitles='{escaped_ass}'[v_out]",
                            "-map", "[v_out]",
                            "-map", "1:a",
                        ])
                    else:
                        cmd.extend(["-map", "0:v", "-map", "1:a"])

                cmd.extend([
                    "-t", f"{slide_dur:.3f}",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "19",
                    "-pix_fmt", "yuv420p",
                    "-r", "30",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-ar", "44100",
                    "-ac", "2",
                    str(seg_out),
                ])

                res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if res.returncode != 0:
                    raise RuntimeError(f"渲染第 {page_num} 页失败: {res.stderr}")

                segment_files.append(seg_out)
                if progress and task_id is not None:
                    progress.advance(task_id)

            # ============================================================
            # 步骤 F: Concat 快速拼合所有切片
            # ============================================================
            if progress and task_id is not None:
                progress.reset(
                    task_id,
                    total=1,
                    description="[magenta][4/4] ⚡ 拼合并混入背景音乐[/magenta]",
                )
            concat_list_file = temp_dir / "concat_list.txt"
            concat_content = "".join(f"file '{p.resolve()}'\n" for p in segment_files)
            concat_list_file.write_text(concat_content, encoding="utf-8")

            merged_video = temp_dir / "merged_no_bgm.mp4"
            cmd_concat = [
                FFMPEG_BIN, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_file),
                "-c", "copy",
                "-movflags", "+faststart",
                str(merged_video),
            ]
            logger.debug("正在执行多页切片快速拼接 (Concat)...")
            res_concat = subprocess.run(cmd_concat, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if res_concat.returncode != 0:
                raise RuntimeError(f"Concat 拼接失败: {res_concat.stderr}")

            # ============================================================
            # 步骤 G: BGM 背景音乐铺底 (可选)
            # ============================================================
            if chosen_music_path and Path(chosen_music_path).is_file():
                logger.debug(f"正在混入背景音乐: {Path(chosen_music_path).name} (音量: {chosen_music_volume})...")
                afade_start = max(0.0, total_video_duration - 2.0)
                cmd_bgm = [
                    FFMPEG_BIN, "-y",
                    "-i", str(merged_video),
                    "-stream_loop", "-1",
                    "-i", str(chosen_music_path),
                    "-filter_complex",
                    f"[1:a]volume={chosen_music_volume:.2f},afade=t=out:st={afade_start:.2f}:d=2[bgm];"
                    f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                    "-map", "0:v",
                    "-map", "[aout]",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-t", f"{total_video_duration:.3f}",
                    "-movflags", "+faststart",
                    str(output_path),
                ]
                res_bgm = subprocess.run(cmd_bgm, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if res_bgm.returncode != 0:
                    logger.warning(f"BGM 混音失败，回退为无 BGM 视频: {res_bgm.stderr}")
                    shutil.move(str(merged_video), str(output_path))
            else:
                shutil.move(str(merged_video), str(output_path))

            if progress and task_id is not None:
                progress.advance(task_id)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"🎉 PDF 视频生成成功!\n"
            f"   - 视频文件: {output_path}\n"
            f"   - 总时长: {total_video_duration:.1f} 秒 ({total_slides} 页)\n"
            f"   - 大小: {size_mb:.2f} MB"
        )

        return {
            "output": str(output_path),
            "srt": str(srt_output_path) if srt_output_path.is_file() else None,
            "duration": round(total_video_duration, 2),
            "slides_count": total_slides,
            "size_mb": round(size_mb, 2),
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
