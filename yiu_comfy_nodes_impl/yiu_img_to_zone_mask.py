import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from .utils import get_current_language, get_localized_tooltips
from .yiu_base_node import YiuBaseNode


class yiuImgToZoneMask(YiuBaseNode):    
    tooltips_dict = {
        "zh-CN": {
            "input": {
                "image": "输入图片（仅用于读取尺寸）",
                "x_pos": "水平位置：0=左边距为0；100=右边距为0",
                "y_pos": "垂直位置：0=上边距为0；100=下边距为0",
                "x_scale": "遮罩宽度占图片宽度的百分比",
                "y_scale": "遮罩高度占图片高度的百分比",
                "lock_yscale_to_xscale": "锁定 y_scale = x_scale（锁定时 y_scale 不参与计算）",
                "radius": "圆角程度：0=直角；50=胶囊（最大圆角）；100=椭圆/圆",
                "feather_radius": "羽化半径：0~100%，映射到(长边或长轴的一半)的像素范围",
                "invert": "反转遮罩：True=外侧为白；False=矩形内为白",
            },
            "output": {
                "mask": "生成的遮罩",
                "help_text": "帮助信息",
            },
        },
        "en": {
            "input": {
                "image": "Input image (used for size only)",
                "x_pos": "Horizontal position: 0=left padding 0; 100=right padding 0",
                "y_pos": "Vertical position: 0=top padding 0; 100=bottom padding 0",
                "x_scale": "Mask width as percentage of image width",
                "y_scale": "Mask height as percentage of image height",
                "lock_yscale_to_xscale": "Lock y_scale = x_scale (y_scale is ignored when locked)",
                "radius": "Corner roundness: 0=sharp; 50=capsule(max); 100=ellipse/circle",
                "feather_radius": "Feather radius: 0~100% mapped to half of the long side/axis",
                "invert": "Invert mask: True=outside white; False=inside white",
            },
            "output": {
                "mask": "Generated mask",
                "help_text": "Help text",
            },
        },
    }

    base_inputs = {
        "image": ("IMAGE", {}),
        "x_pos": ("FLOAT", {"display": "slider", "default": 50.0, "min": 0.0, "max": 100.0, "step": 0.1, "unit": "%"}),
        "y_pos": ("FLOAT", {"display": "slider", "default": 50.0, "min": 0.0, "max": 100.0, "step": 0.1, "unit": "%"}),
        "x_scale": ("FLOAT", {"display": "slider", "default": 50.0, "min": 0.0, "max": 100.0, "step": 0.1, "unit": "%"}),
        "y_scale": (
            "FLOAT",
            {
                "display": "slider",
                "default": 50.0,
                "min": 0.0,
                "max": 100.0,
                "step": 0.1,
                "unit": "%",
                "disabled_when": {"lock_yscale_to_xscale": True},
            },
        ),
        "lock_yscale_to_xscale": ("BOOLEAN", {"default": False}),
        "radius": ("FLOAT", {"display": "slider", "default": 20.0, "min": 0.0, "max": 100.0, "step": 0.1, "unit": "%"}),
        "feather_radius": ("FLOAT", {"display": "slider", "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "unit": "%"}),
        "invert": ("BOOLEAN", {"default": False}),
    }

    base_outputs = {
        "mask": "MASK",
        "help_text": "STRING",
    }

    MAIN = "image_to_zone_mask"

    @classmethod
    def DISPLAY_NAME(cls):
        return "🟦 Image To Zone Mask"

    @classmethod
    def get_help_text(cls):
        lang = get_current_language()
        help_texts = {
            "zh-CN": (
                "📖 使用说明：根据输入 image 的尺寸生成一个圆角矩形遮罩。\n"
                "- x_pos/y_pos：在剩余空隙中的位置比例：left = x_pos/100*(W-maskW)，top 同理。\n"
                "- x_scale/y_scale：遮罩宽高占 image 宽高的百分比。\n"
                "- lock_yscale_to_xscale：True 时 y_scale 不参与计算（UI 侧暂不做联动禁用）。\n"
                "- radius：0=直角；0~50 线性增加到最大圆角（胶囊）；50~100 在胶囊与椭圆之间做插值。\n"
                "- feather_radius：0~100% 映射到 max(maskW,maskH)/2 的像素范围并做高斯模糊。\n"
                "- invert：True 反转遮罩（外侧白），常用于外补/扩图；False 正常遮罩（内部白）。\n"
            ),
            "en": (
                "📖 Generates a rounded-rectangle mask from the input image size.\n"
                "- x_pos/y_pos: position within remaining space: left = x_pos/100*(W-maskW), same for top.\n"
                "- x_scale/y_scale: mask width/height as % of image width/height.\n"
                "- lock_yscale_to_xscale: when True, y_scale is ignored (UI side is not dynamically disabled).\n"
                "- radius: 0=sharp; 0~50 ramps to max corner (capsule); 50~100 blends capsule to ellipse.\n"
                "- feather_radius: 0~100% mapped to max(maskW,maskH)/2 pixels with Gaussian blur.\n"
                "- invert: True inverts mask (outside white); False keeps inside white.\n"
            ),
        }
        return get_localized_tooltips(cls.tooltips_dict, lang) + "\n" + help_texts.get(lang, help_texts["en"])

    @staticmethod
    def _clamp_int(v, low, high):
        return int(max(low, min(high, v)))

    @classmethod
    def _make_roundrect_mask_u8(cls, width, height, left, top, rect_w, rect_h, radius_pct):
        mask_img = Image.new("L", (width, height), 0)
        if rect_w <= 0 or rect_h <= 0:
            return mask_img

        right = left + rect_w
        bottom = top + rect_h
        bbox = (left, top, right, bottom)

        r_max = max(0.0, min(rect_w, rect_h) * 0.5)

        if radius_pct <= 50.0:
            corner_r = (radius_pct / 50.0) * r_max if r_max > 0 else 0.0
            draw = ImageDraw.Draw(mask_img)
            draw.rounded_rectangle(bbox, radius=corner_r, fill=255)
            return mask_img

        capsule = Image.new("L", (width, height), 0)
        if r_max > 0:
            ImageDraw.Draw(capsule).rounded_rectangle(bbox, radius=r_max, fill=255)
        else:
            ImageDraw.Draw(capsule).rectangle(bbox, fill=255)

        ellipse = Image.new("L", (width, height), 0)
        ImageDraw.Draw(ellipse).ellipse(bbox, fill=255)

        t = (radius_pct - 50.0) / 50.0
        return Image.blend(capsule, ellipse, float(max(0.0, min(1.0, t))))

    @classmethod
    def image_to_zone_mask(
        cls,
        image,
        x_pos=50.0,
        y_pos=50.0,
        x_scale=50.0,
        y_scale=50.0,
        lock_yscale_to_xscale=False,
        radius=20.0,
        feather_radius=0.0,
        invert=False,
    ):
        if image.ndim != 4:
            raise ValueError(f"Expected image shape (B,H,W,C), got {tuple(image.shape)}")

        batch_size, height, width, _ = image.shape

        y_scale_eff = x_scale if lock_yscale_to_xscale else y_scale
        rect_w = int(round(width * (float(x_scale) / 100.0)))
        rect_h = int(round(height * (float(y_scale_eff) / 100.0)))

        rect_w = cls._clamp_int(rect_w, 0, width)
        rect_h = cls._clamp_int(rect_h, 0, height)

        space_x = max(0, width - rect_w)
        space_y = max(0, height - rect_h)
        left = int(round((float(x_pos) / 100.0) * space_x)) if space_x > 0 else 0
        top = int(round((float(y_pos) / 100.0) * space_y)) if space_y > 0 else 0

        left = cls._clamp_int(left, 0, space_x)
        top = cls._clamp_int(top, 0, space_y)

        radius_pct = float(max(0.0, min(100.0, radius)))
        mask_img = cls._make_roundrect_mask_u8(width, height, left, top, rect_w, rect_h, radius_pct)

        long_axis_half = (max(rect_w, rect_h) * 0.5) if (rect_w > 0 and rect_h > 0) else (max(width, height) * 0.5)
        feather_px = (float(max(0.0, min(100.0, feather_radius))) / 100.0) * float(long_axis_half)
        if feather_px > 0:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))

        if invert:
            mask_img = ImageOps.invert(mask_img)

        mask_np = np.array(mask_img).astype(np.float32) / 255.0
        mask_np = np.clip(mask_np, 0.0, 1.0)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        if batch_size > 1:
            mask_tensor = mask_tensor.repeat(batch_size, 1, 1)

        return (mask_tensor,)
