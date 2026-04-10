"""去水印菜单"""
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from config import WATERMARK_BRAND_PRESETS
from menus._prompts import confirm_action


def menu_watermark(videos: list[Path]):
    """去水印菜单"""
    from tools.watermark.batch import (
        batch_remove_watermark_delogo,
        batch_remove_watermark_lama,
        batch_remove_watermark_opencv,
    )

    engine = inquirer.select(
        message="选择去水印引擎:",
        choices=[
            Choice("delogo", "⚡ FFmpeg delogo (快速推荐, 适合固定位置 Logo)"),
            Choice("opencv", "🔧 OpenCV (传统算法, 中速)"),
            Choice("lama", "🧠 LaMA (深度学习, 慢, 效果最好)"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
        default="delogo",
    ).execute()

    if engine == "back":
        return

    # 选择水印品牌类型 (预设 / 手动)
    brand_choices = [
        Choice("veo", f"🎬 {WATERMARK_BRAND_PRESETS['veo']['label']}"),
        Choice("custom", "✍️  手动选择/输入坐标"),
        Separator(),
        Choice("back", "⬅️  返回上一级"),
    ]
    brand = inquirer.select(
        message="选择水印品牌类型:",
        choices=brand_choices,
    ).execute()

    if brand == "back":
        return

    # 参考分辨率 (仅在鼠标框选/预设时记录)
    ref_width = 0
    ref_height = 0
    x1 = y1 = x2 = y2 = 0

    if brand == "veo":
        preset = WATERMARK_BRAND_PRESETS["veo"]
        ref_width = preset["ref_width"]
        ref_height = preset["ref_height"]
        x1, y1, x2, y2 = preset["regions"][0]
        print(f"✅ 已选择 Veo 预设区域: {x1},{y1},{x2},{y2} (参考分辨率 {ref_width}x{ref_height})")
    else:
        # 手动输入/框选
        # 使用循环代替递归, 避免栈溢出
        while True:
            print("请输入水印区域坐标: x1,y1,x2,y2")
            print("提示: 输入 's' 或 'select' 可开启鼠标框选 (需本地运行)")
            region_input = inquirer.text(message="区域坐标 (或 s):").execute()

            if region_input.lower() in ["s", "select"]:
                try:
                    import cv2
                    sample_video = videos[0]
                    cap = cv2.VideoCapture(str(sample_video))
                    ref_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    ref_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.2))
                    ret, frame = cap.read()
                    cap.release()

                    if not ret:
                        print("❌ 无法读取视频帧，请手动输入")
                        ref_width = ref_height = 0
                        continue

                    # 缩放帧以适配屏幕 (高分辨率/高DPI 屏幕)
                    max_w, max_h = 1280, 720
                    fh, fw = frame.shape[:2]
                    scale_ratio = min(max_w / fw, max_h / fh, 1.0)
                    if scale_ratio < 1.0:
                        display = cv2.resize(frame, (int(fw * scale_ratio), int(fh * scale_ratio)))
                    else:
                        display = frame

                    print("\n📸 请在弹出的窗口中框选水印区域，按 Enter 或 Space 确认...")
                    cv2.namedWindow("Select Watermark", cv2.WINDOW_NORMAL)
                    x, y, w, h = cv2.selectROI("Select Watermark", display, showCrosshair=True)
                    cv2.destroyAllWindows()

                    # 还原到原始分辨率坐标
                    if scale_ratio < 1.0:
                        x, y, w, h = (
                            x / scale_ratio, y / scale_ratio,
                            w / scale_ratio, h / scale_ratio,
                        )
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                    print(f"✅ 已选择: {x1},{y1},{x2},{y2}")

                    if w == 0 or h == 0:
                        print("⚠️ 未选择区域")
                        ref_width = ref_height = 0
                        continue

                except Exception as e:
                    print(f"❌ 启动图形界面失败: {e}\n请尝试手动输入坐标。")
                    ref_width = ref_height = 0
                    continue
            else:
                try:
                    x1, y1, x2, y2 = [int(p.strip()) for p in region_input.split(',')]
                except (ValueError, TypeError):
                    print("❌ 格式错误，请使用 x1,y1,x2,y2")
                    continue

            # 成功解析坐标, 跳出循环
            break

    if confirm_action(f"确认处理 {len(videos)} 个视频?"):
        if engine == "opencv":
            batch_remove_watermark_opencv(
                videos=videos, regions=[(x1, y1, x2, y2)],
                ref_width=ref_width, ref_height=ref_height,
            )
        elif engine == "delogo":
            batch_remove_watermark_delogo(
                videos=videos, regions=[(x1, y1, x2, y2)],
                ref_width=ref_width, ref_height=ref_height,
            )
        else:
            batch_remove_watermark_lama(
                videos=videos, regions=[(x1, y1, x2, y2)],
                ref_width=ref_width, ref_height=ref_height,
            )
