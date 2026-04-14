import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from config import FFMPEG_BIN, IMAGE_EXTENSIONS, OUTPUT_SLIDESHOW
from tools.common import generate_output_name, logger
from tools.concat.ffmpeg_concat import TRANSITION_PRESETS


def resize_and_pad(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize image with aspect ratio preserved, optionally adding black padding"""
    img_ratio = img.width / img.height
    target_ratio = target_w / target_h

    if img_ratio > target_ratio:
        # scale to target width
        new_w = target_w
        new_h = int(target_w / img_ratio)
    else:
        # scale to target height
        new_h = target_h
        new_w = int(target_h * img_ratio)

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # create black background
    new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(img, (paste_x, paste_y))

    return new_img, paste_x, paste_y, new_w, new_h


def preprocess_image(image_path: Path, output_path: Path, target_w: int, target_h: int):
    img = Image.open(image_path).convert("RGB")
    img, paste_x, paste_y, new_w, new_h = resize_and_pad(img, target_w, target_h)
    img.save(output_path, "JPEG", quality=95)


def discover_slideshow_groups(base_dir: Path, recursive: bool = False) -> list[tuple[Path, list[Path]]]:
    """
    从目录中发现可批量生成幻灯片的图片分组。

    返回:
        [(目录路径, 该目录下图片列表), ...]
    """
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        return []

    if recursive:
        candidate_dirs = sorted({p.parent for p in base_dir.rglob("*") if p.is_file()})
    else:
        candidate_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir())

    groups: list[tuple[Path, list[Path]]] = []
    for directory in candidate_dirs:
        images = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            groups.append((directory, images))
    return groups


def build_texts_from_filenames(image_paths: list[Path]) -> list[str]:
    """
    根据图片文件名自动生成文案:
    - 去扩展名
    - 将下划线/中划线替换为空格
    """
    texts: list[str] = []
    for p in image_paths:
        txt = p.stem.replace("_", " ").replace("-", " ").strip()
        texts.append(txt)
    return texts


def load_texts_from_caption_file(caption_file: Path) -> list[str]:
    """
    从 captions.txt 读取文案（每行一句）。
    """
    if not caption_file.is_file():
        return []

    lines = caption_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def generate_slideshow(
    image_paths: list[Path],
    texts: list[str],
    resolution: tuple[int, int],
    duration_per_image: float = 5.0,
    music_path: Path | str | None = None,
    music_volume: float = 0.3,
    transition: str = "none",
    transition_duration: float = 1.0,
    output_path: Path | str | None = None,
):
    """
    Generate a slideshow from a list of images.
    """
    if not image_paths:
        raise ValueError("No images provided")

    target_w, target_h = resolution
    
    OUTPUT_SLIDESHOW.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_name = generate_output_name("slideshow", ".mp4")
        output_path = OUTPUT_SLIDESHOW / output_name
    output_path = Path(output_path)

    temp_dir = Path(tempfile.mkdtemp(prefix="x-tools_slideshow_"))
    
    try:
        from tools.slideshow.ass_builder import generate_ass_subtitles
        ass_path = temp_dir / "subtitles.ass"
        generate_ass_subtitles(texts, duration_per_image, resolution, ass_path)

        if transition == "none":
            concat_file_content = []
            for i, img_p in enumerate(image_paths):
                out_img = temp_dir / f"{i:04d}.jpg"
                preprocess_image(img_p, out_img, target_w, target_h)
                
                concat_file_content.append(f"file '{out_img.resolve()}'\n")
                concat_file_content.append(f"duration {duration_per_image}\n")
                
            concat_file_content.append(f"file '{temp_dir / f'{(len(image_paths)-1):04d}.jpg'}'\n")

            concat_list_path = temp_dir / "concat_list.txt"
            concat_list_path.write_text("".join(concat_file_content), encoding="utf-8")

            cmd = [
                FFMPEG_BIN, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_path),
            ]

            if music_path:
                cmd.extend(["-i", str(music_path)])

            total_duration = len(image_paths) * duration_per_image

            filter_complex = []
            if music_path:
                afade = f"afade=t=out:st={max(0, total_duration-2):.2f}:d=2"
                filter_complex.append(f"[1:a]volume={music_volume},{afade}[a_out]")
                filter_complex.append(f"[0:v]subtitles='{ass_path.resolve().as_posix()}'[final_v]")
                cmd.extend(["-filter_complex", ";".join(filter_complex)])
                cmd.extend(["-map", "[final_v]", "-map", "[a_out]"])
            else:
                filter_complex.append(f"[0:v]subtitles='{ass_path.resolve().as_posix()}'[final_v]")
                cmd.extend(["-filter_complex", ";".join(filter_complex)])
                cmd.extend(["-map", "[final_v]"])
        else:
            # Transition logic using xfade
            xfade_type = TRANSITION_PRESETS.get(transition, {}).get("xfade")
            td = min(transition_duration, duration_per_image * 0.8) # Ensure transition is not longer than clip

            cmd = [FFMPEG_BIN, "-y"]
            for i, img_p in enumerate(image_paths):
                out_img = temp_dir / f"{i:04d}.jpg"
                preprocess_image(img_p, out_img, target_w, target_h)
                cmd.extend(["-loop", "1", "-t", str(duration_per_image), "-i", str(out_img)])

            music_input_idx = None
            if music_path:
                music_input_idx = len(image_paths)
                cmd.extend(["-i", str(music_path)])

            filter_complex = []
            n = len(image_paths)
            for i in range(n):
                filter_complex.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")

            if n > 1:
                cumulative_offset = duration_per_image - td
                prev_label = "v0"
                for i in range(1, n):
                    out_label = "outv" if i == n - 1 else f"xf{i}"
                    filter_complex.append(
                        f"[{prev_label}][v{i}]xfade=transition={xfade_type}:"
                        f"duration={td}:offset={cumulative_offset:.3f}[{out_label}]"
                    )
                    prev_label = out_label
                    if i < n - 1:
                        cumulative_offset += duration_per_image - td
                total_duration = n * duration_per_image - (n - 1) * td
            else:
                filter_complex.append("[v0]null[outv]")
                total_duration = duration_per_image

            filter_complex.append(f"[outv]subtitles='{ass_path.resolve().as_posix()}'[final_v]")

            if music_path:
                afade = f"afade=t=out:st={max(0, total_duration-2):.2f}:d=2"
                filter_complex.append(f"[{music_input_idx}:a]volume={music_volume},{afade}[a_out]")
                cmd.extend(["-filter_complex", ";".join(filter_complex)])
                cmd.extend(["-map", "[final_v]", "-map", "[a_out]"])
            else:
                cmd.extend(["-filter_complex", ";".join(filter_complex)])
                cmd.extend(["-map", "[final_v]"])

        cmd.extend([
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-t", str(total_duration), 
            "-movflags", "+faststart",
        ])

        if music_path:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])

        cmd.append(str(output_path))

        logger.info(f"Generating slideshow (Transition: {transition})...")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg Error:\n{result.stderr}")
            
    finally:
        for p in temp_dir.glob("*"):
            p.unlink(missing_ok=True)
        temp_dir.rmdir()
        
    logger.info(f"✅ Slideshow generated at {output_path}")
    return {"output": str(output_path)}
