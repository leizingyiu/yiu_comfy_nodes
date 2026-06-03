import torch
import numpy as np
import re
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from nodes import MAX_RESOLUTION
from .utils import get_current_language, build_types,get_localized_tooltips
from .yiu_base_node import YiuBaseNode

print('this is yiu_img_cnv_placer.py - 202507051718')

class yiuImgCnvPlacer(YiuBaseNode):    
    tooltips_dict={
        "zh-CN":{ 
            "input":{
                "image": "需要扩图的图片",
                "mask": "需要扩图图片上绘制的蒙版",
                "resolution": "SDXL与FLUX分辨率，用作输出图片的画板尺寸",
                "width_overwrite": "宽度覆盖，当使用非上述分辨率时，将覆盖分辨率的宽度。",
                "height_overwrite": "高度覆盖，当使用非上述分辨率时，将覆盖分辨率的高度。",
                "x_pos": "X轴位置，图片在画板的最左边还是最右边",
                "y_pos": "Y轴位置，图片在画板的最上边还是最下边",
                "scale": "缩放，图片相对画板的占比",
                "scale_ref_canvas": "缩放参考：true以画布为参考（100%=适配画布），false以自身为参考（100%=原尺寸）",
                "hex_color": "带井号的HEX色值，例如 #000000",
                "feather_radius": "羽化半径，扩图时羽化的范围",
            },
            "output":{
                "image": "已经放在扩展画板中的图像",
                "original_mask": "已经放在扩展画板中的原来的遮罩",
                "expand_mask": "扩图节点需要的扩展遮罩",
                "bg_color_image": "以hex色值为背景色合成后的输出图像",
                "edge_expand_image": "以图片边缘向外放射扩展合成后的输出图像",
                "help_text": "帮助信息",
            }
        },
        "en":{
            "input": {
                "image": "Image to be expanded",
                "mask": "Mask to be drawn on the image to be expanded",
                "resolution": "SDXL and FLUX resolution, used as the canvas size for output images",
                "width_overwrite": "Width override, when using a resolution other than the ones above, will override the width of the resolution.",
                "height_overwrite": "Height override, when using a resolution other than the above, will override the resolution's height.",
                "x_pos": "X-axis position, is the image on the left or right of the canvas",
                "y_pos": "Y-axis position, is the image on the top or bottom of the canvas",
                "scale": "Scale, the proportion of the image relative to the canvas",
                "scale_ref_canvas": "Scale reference: true=canvas (100% fits canvas), false=self (100% keeps original size)",
                "hex_color": "HEX color with #, e.g. #000000",
                "feather_radius": "Feather radius, the range of feathering when expanding the image",
            },
            "output":{
                "image": "The image already placed in the extended artboard",
                "original_mask": "The original mask already placed in the extended artboard",
                "expand_mask": "The expanded mask required by the expansion node",
                "bg_color_image": "Composited image with the HEX background color",
                "edge_expand_image": "Composited image with radial edge-extended background",
                "help_text": "Help Text",
            }
        }
    }


    base_inputs = {
                "image": ("IMAGE",{}),
                "mask": ("MASK",{"default": None}),
                "resolution": (
                    [
                        "1600x1024",
                        "1536x1536",
                        "1536x1024",
                        "1344x768",
                        "1280x768",
                        "1152x896",
                        "1152x768",
                        "1104x832",
                        "1024x1600",
                        "1024x1536",
                        "1024x1344",
                        "1024x1280",
                        "1024x1152",
                        "1024x1024",
                        "960x1344",
                        "896x1536",
                        "896x1152",
                        "896x768",
                        "832x1104",
                        "768x1536",
                        "768x1280",
                        "768x1152",
                        "768x1024",
                        "768x896"
                    ],
                    {"default": "1024x1024","unit":"px"}
                ),

                "width_overwrite": ("INT", {"default": 0, "min": 0,"unit":"px"}),
                "height_overwrite": ("INT", {"default": 0, "min": 0,"unit":"px"}),
                "x_pos": ("FLOAT", {"display": "slider", "default": 50, "min": 0.0, "max": 100.0,"unit":"%"}),
                "y_pos": ("FLOAT", {"display": "slider", "default": 50, "min": 0.0, "max": 100.0,"unit":"%"}),
          
                "scale": ("FLOAT", {"display": "slider", "default":100, "min": 1, "max": 100, "step": 0.1 ,"unit":"%"}),
                "scale_ref_canvas": ("BOOLEAN", {"default": True}),
                "hex_color": ("STRING", {"default": "#000000"}),
                "feather_radius": ("INT", {"default": 20, "min": 0, "max": MAX_RESOLUTION, "step": 1 ,"unit":"px"}),
            }
    
    base_outputs = {
            "image": "IMAGE",
            "original_mask": "MASK",
            "expand_mask": "MASK",
            "bg_image": "IMAGE",
            "edge_expand_image": "IMAGE",
            "help_text": "STRING"
        }
    
  


    MAIN='better_image_pad_for_outpainting'



    @classmethod
    def DISPLAY_NAME(cls):
        return "🖊️ Image Canvas Placer"
    
    @classmethod
    def get_help_text(self):
        lang = get_current_language()
        help_texts = {
            "zh-CN": '''📖 使用说明：将输入的image 、 mask，根据x_pos（水平方向的位置）、y_pos（垂直方向的位置）、scale（在画板中的占比），放置在 resolution（画布分辨率）大小的画板中。
            输入的image建议先经过抠图节点处理。
            输出的image 和 expand_mask 可等效于“外补画板”，接到扩图工作流中。
            输出的 original_mask 为对应缩放移动的输入的mask，可对应接到Lama，或者接到我的 yiu_fit_itm_2_msk 节点的 camvas_mask 。''',
            "en": "📖 Recommended to cut out the product image first. Mask decides product placement."
        }
        return get_localized_tooltips(self.tooltips_dict,lang)+'\n'+help_texts.get(lang, help_texts["en"])
    
    @classmethod
    def adjust_levels(self, image, black=0, white=1, gamma=1.0):
        """调整图像色阶，类似Photoshop的色阶功能"""
        # 将图像转换为numpy数组
        arr = np.array(image, dtype=np.float32) / 255.0
        
        # 应用色阶调整
        arr = np.clip((arr - black) / (white - black), 0, 1)
        arr = np.power(arr, gamma)
        
        # 转换回PIL图像
        return Image.fromarray((arr * 255).astype(np.uint8))
    
    @classmethod
    def better_image_pad_for_outpainting(self, image, mask=None, resolution="1024x1024", width_overwrite=0, height_overwrite=0, x_pos=50, y_pos=50, scale=100, scale_ref_canvas=True, hex_color="#000000", feather_radius=20):        
        # 解析 SDXL 分辨率
        try:
            sdxl_width, sdxl_height = map(int, resolution.split("x"))
        except ValueError:
            raise ValueError(f"Invalid SDXL resolution format: {resolution}")

        width = width_overwrite if width_overwrite != 0 else sdxl_width
        height = height_overwrite if height_overwrite != 0 else sdxl_height

        # 处理图像
        batch_size, img_h, img_w, channels = image.shape
        image_np = image[0].cpu().numpy() * 255.0
        image_np = image_np.astype(np.uint8)
        
        # 转换为RGBA格式以支持透明
        if channels == 3:
            pil_image = Image.fromarray(image_np).convert("RGBA")
        else:
            pil_image = Image.fromarray(image_np)

        expanded_width = width
        expanded_height = height
        new_width,new_height =  map(int, self.scale_rect(img_w, img_h, expanded_width, expanded_height, scale, scale_ref_canvas))

        # 缩小图像
        resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        x = int(x_pos / 100 * (expanded_width - new_width))
        y = int(y_pos / 100 * (expanded_height - new_height))
        
        print("x:", x, "y:", y)
        print("new_width:", new_width, "new_height:", new_height)
        print("expanded_width:", expanded_width, "expanded_height:", expanded_height)


        
        # 创建透明背景的新图像
        expanded_image = Image.new("RGBA", (expanded_width, expanded_height), (0, 0, 0, 0))
        
        # 将缩小后的图像粘贴到中心
        paste_position = (x, y)

        expanded_image.paste(resized_image, paste_position)

        r, g, b = self.parse_hex_color(hex_color)
        bg_image = Image.new("RGBA", (expanded_width, expanded_height), (r, g, b, 255))
        bg_image = Image.alpha_composite(bg_image, expanded_image)

        edge_expand_image = self.edge_expand_composite(expanded_image, resized_image, paste_position, expanded_width, expanded_height, (r, g, b))
        
        # 创建扩展区域的蒙版
        expand_mask = Image.new("L", (expanded_width, expanded_height), 0)
        
        # 创建中心区域（非扩展区域）的矩形
        img_box = (
            x, 
            y, 
            x + new_width, 
            y + new_height
        )
        
        # 在扩展蒙版上绘制白色矩形（中心区域）
        draw = Image.new("L", (new_width, new_height), 255)
        expand_mask.paste(draw, img_box)
        
        # 反转蒙版得到扩展区域蒙版
        expand_mask = ImageOps.invert(expand_mask)
        
        # 应用羽化效果
        if feather_radius > 0:
            # 先应用两倍模糊量
            expand_mask = expand_mask.filter(ImageFilter.GaussianBlur(feather_radius * 2))
            
            # 使用色阶调整：将0.5灰色变为1白色，0黑色保持不变
            expand_mask = self.adjust_levels(expand_mask, black=0, white=0.5, gamma=1.0)
        
        # 将PIL图像转换回张量
        expanded_image_np = np.array(expanded_image).astype(np.float32) / 255.0
        expanded_image_tensor = torch.from_numpy(expanded_image_np).unsqueeze(0)

        bg_color_image_np = np.array(bg_image).astype(np.float32) / 255.0
        bg_color_image_tensor = torch.from_numpy(bg_color_image_np).unsqueeze(0)

        edge_expand_image_np = np.array(edge_expand_image).astype(np.float32) / 255.0
        edge_expand_image_tensor = torch.from_numpy(edge_expand_image_np).unsqueeze(0)
        
        # 处理原始蒙版（如果有）
        original_mask_tensor = None
        if mask is not None:
            # 将蒙版张量转换为PIL图像
            mask_np = mask[0].cpu().numpy() * 255.0
            mask_np = mask_np.astype(np.uint8)
            pil_mask = Image.fromarray(mask_np)
            
            # 缩小蒙版
            resized_mask = pil_mask.resize((new_width, new_height), Image.LANCZOS)
            
            # 创建新蒙版（黑色背景）
            expanded_original_mask = Image.new("L", (expanded_width, expanded_height), 0)
            
            # 将缩小后的蒙版粘贴到中心
            expanded_original_mask.paste(resized_mask, paste_position)
            
            # 将PIL蒙版转换回张量
            expanded_original_mask_np = np.array(expanded_original_mask).astype(np.float32) / 255.0
            original_mask_tensor = torch.from_numpy(expanded_original_mask_np).unsqueeze(0)
        
        # 将扩展蒙版转换为张量
        expand_mask_np = np.array(expand_mask).astype(np.float32) / 255.0
        expand_mask_tensor = torch.from_numpy(expand_mask_np).unsqueeze(0)
        
        # 如果输入是批量处理，复制结果以匹配批量大小
        if batch_size > 1:
            expanded_image_tensor = expanded_image_tensor.repeat(batch_size, 1, 1, 1)
            expand_mask_tensor = expand_mask_tensor.repeat(batch_size, 1, 1)
            bg_color_image_tensor = bg_color_image_tensor.repeat(batch_size, 1, 1, 1)
            edge_expand_image_tensor = edge_expand_image_tensor.repeat(batch_size, 1, 1, 1)
            if original_mask_tensor is not None:
                original_mask_tensor = original_mask_tensor.repeat(batch_size, 1, 1)
        
        # 返回结果，如果没有原始蒙版则返回全零蒙版
        if original_mask_tensor is not None:
            return (expanded_image_tensor, original_mask_tensor, expand_mask_tensor, bg_color_image_tensor, edge_expand_image_tensor,)
        else:
            empty_mask = torch.zeros((batch_size, expanded_height, expanded_width))
            return (expanded_image_tensor, empty_mask, expand_mask_tensor, bg_color_image_tensor, edge_expand_image_tensor,)


    @classmethod
    def scale_rect(self, width, height, container_width, container_height, scale, scale_ref_canvas=True):
        """计算缩放后的矩形尺寸"""
        if scale == 0:
            return (0, 0)
        if scale_ref_canvas:
            fit_scale = min(container_width / width, container_height / height)
            scale_factor = fit_scale * (scale / 100)
        else:
            scale_factor = scale / 100
        return (width * scale_factor, height * scale_factor)
    
    @classmethod
    def parse_hex_color(cls, hex_color):
        match = re.fullmatch(r"#([0-9A-Fa-f]{6})", (hex_color or "").strip())
        if not match:
            raise ValueError(f"Invalid hex color: {hex_color} (expected like #RRGGBB)")
        value = match.group(1)
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        return r, g, b
    
    @classmethod
    def rgb_to_hsv_np(cls, rgb):
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        delta = maxc - minc
        s = np.where(maxc == 0, 0.0, delta / np.maximum(maxc, 1e-8))

        h = np.zeros_like(maxc, dtype=np.float32)
        mask = delta > 1e-8
        rc = np.where(mask, (maxc - r) / np.maximum(delta, 1e-8), 0.0)
        gc = np.where(mask, (maxc - g) / np.maximum(delta, 1e-8), 0.0)
        bc = np.where(mask, (maxc - b) / np.maximum(delta, 1e-8), 0.0)

        is_r = (r == maxc) & mask
        is_g = (g == maxc) & mask
        is_b = (b == maxc) & mask

        h = np.where(is_r, (bc - gc), h)
        h = np.where(is_g, 2.0 + (rc - bc), h)
        h = np.where(is_b, 4.0 + (gc - rc), h)
        h = (h / 6.0) % 1.0
        return np.stack([h.astype(np.float32), s.astype(np.float32), v.astype(np.float32)], axis=-1)
    
    @classmethod
    def hsv_to_rgb_np(cls, hsv):
        h = hsv[..., 0] % 1.0
        s = np.clip(hsv[..., 1], 0.0, 1.0)
        v = np.clip(hsv[..., 2], 0.0, 1.0)

        i = np.floor(h * 6.0).astype(np.int32)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)

        i_mod = i % 6
        r = np.zeros_like(v, dtype=np.float32)
        g = np.zeros_like(v, dtype=np.float32)
        b = np.zeros_like(v, dtype=np.float32)

        r = np.where(i_mod == 0, v, r)
        g = np.where(i_mod == 0, t, g)
        b = np.where(i_mod == 0, p, b)

        r = np.where(i_mod == 1, q, r)
        g = np.where(i_mod == 1, v, g)
        b = np.where(i_mod == 1, p, b)

        r = np.where(i_mod == 2, p, r)
        g = np.where(i_mod == 2, v, g)
        b = np.where(i_mod == 2, t, b)

        r = np.where(i_mod == 3, p, r)
        g = np.where(i_mod == 3, q, g)
        b = np.where(i_mod == 3, v, b)

        r = np.where(i_mod == 4, t, r)
        g = np.where(i_mod == 4, p, g)
        b = np.where(i_mod == 4, v, b)

        r = np.where(i_mod == 5, v, r)
        g = np.where(i_mod == 5, p, g)
        b = np.where(i_mod == 5, q, b)

        return np.stack([r, g, b], axis=-1)
    
    @classmethod
    def hue_mean_np(cls, h1, h2):
        a1 = h1 * (2.0 * np.pi)
        a2 = h2 * (2.0 * np.pi)
        x = (np.cos(a1) + np.cos(a2)) * 0.5
        y = (np.sin(a1) + np.sin(a2)) * 0.5
        a = np.arctan2(y, x)
        return (a % (2.0 * np.pi)) / (2.0 * np.pi)
    
    @classmethod
    def edge_expand_composite(cls, expanded_image, resized_image, paste_position, expanded_width, expanded_height, fallback_rgb=(0, 0, 0)):
        rgba = np.array(resized_image.convert("RGBA"), dtype=np.float32) / 255.0
        alpha = rgba[..., 3] > (1.0 / 255.0)
        edge = alpha.copy()
        if alpha.shape[0] > 2 and alpha.shape[1] > 2:
            interior = np.zeros_like(alpha, dtype=bool)
            interior[1:-1, 1:-1] = (
                alpha[1:-1, 1:-1]
                & alpha[0:-2, 1:-1]
                & alpha[2:, 1:-1]
                & alpha[1:-1, 0:-2]
                & alpha[1:-1, 2:]
            )
            edge = alpha & (~interior)

        ys, xs = np.where(edge)
        if ys.size == 0:
            h, w = alpha.shape
            xs_top = np.arange(w)
            ys_top = np.zeros(w, dtype=np.int32)
            xs_bottom = np.arange(w)
            ys_bottom = np.full(w, h - 1, dtype=np.int32)
            ys_left = np.arange(h)
            xs_left = np.zeros(h, dtype=np.int32)
            ys_right = np.arange(h)
            xs_right = np.full(h, w - 1, dtype=np.int32)
            xs = np.concatenate([xs_top, xs_bottom, xs_left, xs_right])
            ys = np.concatenate([ys_top, ys_bottom, ys_left, ys_right])

        rgb_edge = rgba[ys, xs, :3]
        hsv_edge = cls.rgb_to_hsv_np(rgb_edge)

        x0, y0 = paste_position
        cx = x0 + (rgba.shape[1] / 2.0)
        cy = y0 + (rgba.shape[0] / 2.0)
        bx = x0 + xs.astype(np.float32)
        by = y0 + ys.astype(np.float32)
        dx = bx - cx
        dy = by - cy
        angles = (np.arctan2(dy, dx) % (2.0 * np.pi)).astype(np.float32)
        dist = np.hypot(dx, dy).astype(np.float32)

        step = (2.0 * np.pi) / 720.0
        bins = np.floor(angles / step).astype(np.int32)
        order = np.argsort(bins, kind="stable")
        bins_sorted = bins[order]
        angles_sorted = angles[order]
        dist_sorted = dist[order]
        hsv_sorted = hsv_edge[order]

        unique_angles = []
        unique_hsv = []
        i = 0
        n = bins_sorted.size
        while i < n:
            j = i + 1
            while j < n and bins_sorted[j] == bins_sorted[i]:
                j += 1
            group = slice(i, j)
            k = i + int(np.argmax(dist_sorted[group]))
            unique_angles.append(float(angles_sorted[k]))
            unique_hsv.append(hsv_sorted[k])
            i = j

        sample_angles = np.array(unique_angles, dtype=np.float32)
        sample_hsv = np.array(unique_hsv, dtype=np.float32)
        if sample_angles.size < 2:
            r, g, b = fallback_rgb
            bg = Image.new("RGBA", (expanded_width, expanded_height), (int(r), int(g), int(b), 255))
            return Image.alpha_composite(bg, expanded_image)

        sort_idx = np.argsort(sample_angles)
        sample_angles = sample_angles[sort_idx]
        sample_hsv = sample_hsv[sort_idx]

        yy, xx = np.mgrid[0:expanded_height, 0:expanded_width]
        ang_map = (np.arctan2(yy.astype(np.float32) - cy, xx.astype(np.float32) - cx) % (2.0 * np.pi)).astype(np.float32)
        flat = ang_map.reshape(-1)
        idx = np.searchsorted(sample_angles, flat, side="left").astype(np.int32)
        n_samples = sample_angles.size
        right = idx % n_samples
        left = (idx - 1) % n_samples

        h_l = sample_hsv[:, 0].take(left)
        h_r = sample_hsv[:, 0].take(right)
        s_l = sample_hsv[:, 1].take(left)
        s_r = sample_hsv[:, 1].take(right)
        v_l = sample_hsv[:, 2].take(left)
        v_r = sample_hsv[:, 2].take(right)

        h = cls.hue_mean_np(h_l, h_r)
        s = (s_l + s_r) * 0.5
        v = (v_l + v_r) * 0.5
        hsv_bg = np.stack([h, s, v], axis=-1).reshape(expanded_height, expanded_width, 3)
        rgb_bg = cls.hsv_to_rgb_np(hsv_bg)
        rgb_u8 = np.clip(rgb_bg * 255.0, 0, 255).astype(np.uint8)
        a_u8 = np.full((expanded_height, expanded_width, 1), 255, dtype=np.uint8)
        bg_rgba = np.concatenate([rgb_u8, a_u8], axis=-1)
        bg_image = Image.fromarray(bg_rgba, mode="RGBA")

        return Image.alpha_composite(bg_image, expanded_image)
    
