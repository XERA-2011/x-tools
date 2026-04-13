"""
歌词 MV 渲染器

使用 Pillow 逐帧生成动感歌词图像
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tools.mv.beat_detector import BeatInfo
from tools.mv.parser import LyricLine


class LyricRenderer:
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        highlight_color: str = "#FF3333",
        glow: bool = True,
        bg_image_path: Path | None = None,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.highlight_color = highlight_color
        self.glow = glow
        self.bg_image_path = bg_image_path
        
        self.bg_image = None
        if self.bg_image_path and Path(self.bg_image_path).is_file():
            try:
                import PIL.ImageEnhance
                import PIL.ImageOps
                bg_img = Image.open(self.bg_image_path).convert("RGBA")
                self.bg_image = PIL.ImageOps.fit(bg_img, (self.width, self.height), Image.Resampling.LANCZOS)
                # 降低背景亮度以凸显歌词
                enhancer = PIL.ImageEnhance.Brightness(self.bg_image)
                self.bg_image = enhancer.enhance(0.4)
            except Exception as e:
                print(f"Warning: Failed to load background image: {e}")
        
        # 字体路径 (优先使用系统粗体，或者回退到默认)
        self.font_path = self._get_default_font()
        
        # 基础字号计算
        self.base_font_size = int(min(width, height) * 0.12)
        
    def _get_default_font(self) -> str:
        import os
        import platform
        system = platform.system()
        paths = []
        if system == "Darwin":
            paths = [
                "/Library/Fonts/Impact.ttf",
                "/Library/Fonts/Arial Black.ttf",
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/HelveticaNeue.ttc",
                "/Library/Fonts/Arial Bold.ttf"
            ]
        elif system == "Windows":
            paths = [
                "C:\\Windows\\Fonts\\msyhbd.ttc",  # 微软雅黑粗体
                "C:\\Windows\\Fonts\\simhei.ttf",
                "C:\\Windows\\Fonts\\ariblk.ttf"
            ]
        elif system == "Linux":
            paths = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
            ]
            
        for p in paths:
            if os.path.exists(p):
                return p
        print("Warning: Default fonts not found, using PIL default")
        return "" # 回退到 PIL 默认 (不支持中文)
        
    def _create_glow(self, image: Image.Image, radius: int = 10) -> Image.Image:
        """为图像创建发光效果"""
        glow = image.filter(ImageFilter.GaussianBlur(radius))
        # 增强亮度
        return glow

    def render_frame(
        self,
        time: float,
        lyrics: list[LyricLine],
        beat_info: BeatInfo,
    ) -> Image.Image:
        """渲染单帧图像"""
        # 创建背景
        if self.bg_image:
            img = self.bg_image.copy()
        else:
            img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 255))
        # 获取当前歌词和前后歌词用于过渡
        current_line = None
        for line in lyrics:
            if line.start_time <= time <= line.end_time:
                current_line = line
                break
                
        if not current_line:
            # 在歌曲开头或中间间奏，不显示主词，但可能有轻微的背景粒子动画 (Todo)
            return img.convert("RGB")

        # --- 1. 计算节拍驱动的动画参数 ---
        # 缩放比例: 保持 1.0 固定，不缩放，消除所有不必要的抖动抽搐
        scale = 1.0
        
        # 音量包络驱动的透明度或色散
        volume = beat_info.get_volume_at(time)
        opacity = int(255 * (0.8 + volume * 0.2))
        
        # 进退场淡入淡出动画
        fade_alpha = 1.0
        fade_time = 0.3
        time_since_start = time - current_line.start_time
        time_till_end = current_line.end_time - time
        
        if time_since_start < fade_time:
            fade_alpha = time_since_start / fade_time
        elif time_till_end < fade_time:
            fade_alpha = time_till_end / fade_time
            
        final_alpha = int(opacity * fade_alpha)

        # --- 2. 绘制歌词层 ---
        text_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)
        
        try:
            # 缩放字号
            current_font_size = int(self.base_font_size * scale)
            font = ImageFont.truetype(self.font_path, current_font_size) if self.font_path else ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()

        # 计算空格的宽度
        s_left, _, s_right, _ = text_draw.textbbox((0,0), " ", font=font)
        space_w = s_right - s_left if s_right > s_left else current_font_size * 0.3
        
        max_line_width = self.width * 0.85
        lines_to_draw = []
        current_visual_line = []
        current_visual_w = 0
        
        def is_ascii(s):
            return all(ord(c) < 128 for c in s)

        # 智能折行与排布
        for word in current_line.words:
            # 保持正常词汇大小写，去除结尾常见的逗号、句号等无意义标点，使画面更干净
            display_text = word.text.strip().strip(",.?!;:，。？！；：")
            
            # 如果剔除空格后没有内容了(比如本身就是断点)，则跳过
            if not display_text:
                continue

            # 过滤掉常见的无意义间奏提示词，例如 [Music], (music), ♪
            lower_text = display_text.lower()
            if lower_text == 'music' or '♪' in lower_text or lower_text.startswith('[') or lower_text.endswith(']'):
                continue
                
            w_left, w_top, w_right, w_bottom = text_draw.textbbox((0, 0), display_text, font=font)
            word_w = w_right - w_left
            word_h = w_bottom - w_top
            
            needs_space = is_ascii(display_text)
            
            # 是否需要换行
            if current_visual_line and (current_visual_w + space_w + word_w > max_line_width):
                lines_to_draw.append({
                    "words": current_visual_line,
                    "width": current_visual_w - (space_w if current_visual_line[-1]["needs_space"] else 0),
                    "height": word_h
                })
                current_visual_line = []
                current_visual_w = 0
                
            current_visual_line.append({
                "obj": word,
                "text": display_text,
                "w": word_w,
                "h": word_h,
                "needs_space": needs_space
            })
            current_visual_w += word_w + (space_w if needs_space else 0)

        if current_visual_line:
            lines_to_draw.append({
                "words": current_visual_line,
                "width": current_visual_w - (space_w if current_visual_line[-1]["needs_space"] else 0),
                "height": current_visual_line[0]["h"]
            })

        # 计算多行文本的总高度，以便居中
        line_spacing = current_font_size * 0.2
        if not lines_to_draw:
            return img.convert("RGB")
            
        total_h = sum(line["height"] for line in lines_to_draw) + line_spacing * (len(lines_to_draw) - 1)
        start_y = (self.height - total_h) / 2
        

        for row in lines_to_draw:
            start_x = (self.width - row["width"]) / 2
            current_x = start_x
            
            for item in row["words"]:
                word_obj = item["obj"]
                
                # 状态判定 (是否有滞后或者延长)
                is_active = False
                if word_obj.start_time - 0.1 <= time <= word_obj.end_time + 0.1:
                    is_active = True
                    
                color = self.highlight_color if is_active else (255, 255, 255)
                if isinstance(color, str):
                    from PIL import ImageColor
                    color = ImageColor.getrgb(color)
                color_with_alpha = (*color[:3], final_alpha)
                
                shake_x = 0
                shake_y = 0
                
                text_draw.text(
                    (current_x + shake_x, start_y + shake_y), 
                    item["text"], 
                    font=font, 
                    fill=color_with_alpha,
                    stroke_width=int(scale * 2), # 增加轻描黑边提升质感
                    stroke_fill=(0, 0, 0, final_alpha)
                )
                
                current_x += item["w"] + (space_w if item["needs_space"] else 0)
                
            start_y += row["height"] + line_spacing

        # --- 3. 合成图层 ---
        if self.glow:
            glow_layer = self._create_glow(text_layer, radius=int(15 * scale))
            img.paste(glow_layer, (0, 0), glow_layer)
            
        img.paste(text_layer, (0, 0), text_layer)
        
        # 移除了所有的 Glitch (轻微通道色偏移) 和画面随机抽搐抖动，实现完全平稳的纯享版发光字幕

        return img.convert("RGB")
