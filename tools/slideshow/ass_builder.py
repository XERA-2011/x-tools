from pathlib import Path


def format_ass_time(sec: float) -> str:
    """Format time in seconds to ASS time format: H:MM:SS.cs"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int((sec - int(sec)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def wrap_text_into_tokens(txt: str, max_chars_per_line: int) -> list[str]:
    """
    Wrap text by logically inserting ASS newline \\N manually.
    """
    tokens = []
    current_len = 0
    max_units = max_chars_per_line * 2
    for char in txt:
        char_w = 2 if ord(char) > 255 else 1
        if current_len + char_w > max_units:
            tokens.append("\\N")
            current_len = 0
        tokens.append(char)
        current_len += char_w
    return tokens

def generate_ass_subtitles(
    texts: list[str],
    slide_duration: float,
    resolution: tuple[int, int],
    output_path: Path
):
    """
    Generate an ASS subtitle file for the slideshow, providing a typewriter effect.
    """
    width, height = resolution
    
    # Base styling calculations similar to what Pillow used:
    font_size = max(height // 20, 24)
    stroke_width = max(2, font_size // 15)
    margin_v = int(height * 0.1)
    margin_h = int(width * 0.1)

    # Note: PrimaryColour &H0000FFFF is Yellow (A, B, G, R)
    format_line = (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
    )
    style_line = (
        f"Style: Default,STHeiti Light,{font_size},&H0000FFFF,&H000000FF,&H00000000,"
        f"&H00000000,-1,0,0,0,100,100,0,0,1,{stroke_width},0,2,{margin_h},{margin_h},{margin_v},1"
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
    max_chars_per_line = max(1, int(available_width / font_size))
    
    events = []
    
    for i, txt in enumerate(texts):
        if not txt.strip():
            continue
            
        tokens = wrap_text_into_tokens(txt, max_chars_per_line)
        text_len = len(tokens)
            
        slide_start = i * slide_duration
        slide_end = slide_start + slide_duration
        
        # Typewriter speed: ideally 0.1s per char, but cap at 80% of slide duration
        ideal_type_duration = text_len * 0.1
        actual_type_duration = min(ideal_type_duration, slide_duration * 0.8)
        char_speed = actual_type_duration / text_len if text_len > 0 else 0
        
        for char_idx in range(text_len + 1):
            evt_start = slide_start + char_idx * char_speed
            
            if char_idx < text_len:
                evt_end = slide_start + (char_idx + 1) * char_speed
                # Insert the invisible tag at the exact character boundary
                line_text = "".join(tokens[:char_idx]) + "{\\alpha&HFF&}" + "".join(tokens[char_idx:])
            else:
                # Last segment holds until the slide ends
                evt_end = slide_end
                line_text = "".join(tokens)
                
            # If the event has no duration, skip it
            if evt_end - evt_start <= 0.001:
                continue
                
            start_str = format_ass_time(evt_start)
            end_str = format_ass_time(evt_end)
            
            # Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
            events.append(
                f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{line_text}"
            )

    ass_content = ass_header + "\n".join(events) + "\n"
    output_path.write_text(ass_content, encoding="utf-8")
    return output_path
