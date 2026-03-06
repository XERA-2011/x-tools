import re

def scale_style(style_line, target_height):
    parts = style_line.split(":")
    name = parts[0]
    values = parts[1].split(",")
    
    # 按照 1080p 作为基准
    scale = target_height / 1080.0
    
    # Fontsize is index 2
    fontsize = float(values[2])
    # Base size in my tool was 22, let's make it bigger by default (e.g. base 45)
    base_fontsize = 45 
    values[2] = str(int(base_fontsize * scale))
    
    # MarginV is index 21
    # Base MarginV was 30, let's move it up to e.g. 25th percentile for vertical videos!
    # For a vertical video, maybe MarginV = 0.2 * target_height
    # For horizontal, MarginV = 0.05 * target_height
    is_vertical = target_height > 1200
    if is_vertical:
        target_margin_v = int(target_height * 0.22) # Move up about 22% of screen height
    else:
        target_margin_v = int(target_height * 0.08)

    values[21] = str(target_margin_v)
    
    # Also adjust MarginL and MarginR to be e.g. 5% of height?
    margin_lr = int(target_height * 0.05)
    values[19] = str(margin_lr)
    values[20] = str(margin_lr)
    
    # Also scale Outline and Shadow?
    # Outline is index 16
    values[16] = str(int(float(values[16]) * scale * 1.5)) # make outline slightly thicker
    # Shadow is index 17
    values[17] = str(int(float(values[17]) * scale * 1.5))

    return f"{name}:{','.join(values)}"

print(scale_style("Style: Default,Arial,22,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1", 1920))
