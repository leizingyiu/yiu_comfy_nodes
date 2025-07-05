import torch
import numpy as np
from PIL import Image, ImageOps
from .utils import get_current_language, build_types,get_localized_tooltips
from .yiu_base_node import YiuBaseNode

print("this is yiu_fit_item_to_mask.py - 202507051718")

class yiuFitItm2Msk(YiuBaseNode):
    

    tooltips_dict = {
        "zh-CN":{"input": {
            "item_image": "产品图片，可先经过抠图节点处理后接入",
            "item_mask": "产品图片遮罩，可从抠图节点获取",
            "canvas_image": "背景图片，建议使用场景参考图片",
            "canvas_mask": "背景遮罩，产品需要放在背景中的什么位置，就在背景遮罩中绘制",
            "object_fit": "产品在遮罩尺寸中，填充或适应",
            "bg_color": "输出时使用的背景颜色，默认黑色",
        },
        "output":{
            "item_on_transparent":"调整好位置和缩放的产品，在透明背景上",
              "item_mask":"调整好位置和缩放的产品遮罩",
                "item_on_bgcolor":"调整好位置和缩放的产品遮罩，在背景颜色上",
                  "item_on_canvas":"调整好位置和缩放的产品遮罩，在原背景图片上",
                  "help_text":"帮助信息"
        }},
        "en":{"input": {
            "item_image": "Product image, recommended to process with a cutout node first",
            "item_mask": "Product image mask, can be obtained from the cutout node",
            "canvas_image": "Background image, suggested to use a scene reference image",
            "canvas_mask": "Background mask, draw where the product should be placed in the background",
            "object_fit": "How the product fits inside the mask area: fill or fit",
            "bg_color": "Background color for output, defaults to black"
        },
        "output":{
            "item_on_transparent":"Adjust the position and scale of the product, on a transparent background",
            "item_mask":"Adjust the position and scale of the product mask",
            "item_on_bgcolor":"Adjust the position and scale of the product mask, on the background color",
            "item_on_canvas":"Adjust the position and scale of the product mask, on the original background image",
            "help_text":"Help Text"
        }
        }
    }

    base_inputs = {
            "item_image": ("IMAGE", {}),
            "item_mask": ("MASK", {}),
            "canvas_image": ("IMAGE", {}),
            "canvas_mask": ("MASK", {}),
            "object_fit": (["fill", "fit"], {"default": "fit"}),
            "bg_color": ("STRING", {
                "default": "#000000", 
                "pattern": "^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$"
            }),
        }

    base_outputs={
        "item_on_transparent":"IMAGE",
         "item_mask":"MASK",
         "item_on_bgcolor":"IMAGE",
         "item_on_canvas":"IMAGE",
        "help_text":"STRING"
    }


    MAIN='yiu_fit_item_to_mask'

    @classmethod
    def DISPLAY_NAME(cls):
        return "🖊️ Yiu Fit Item To Mask"
    
    @classmethod
    def get_help_text(self):
        lang = get_current_language()
        help_texts = {
            "zh-CN": "📖 将图1(item)物品放置在图2(canvas)的遮罩中。物品(item)图片建议先抠图，遮罩(canvas_mask)决定物品位置。",
            "en": "📖 Place the item in Figure 1 (item) in the mask in Figure 2 (canvas). It is recommended to cut out the item image first, and the mask (canvas_mask) determines the position of the item."
        }
        return  get_localized_tooltips(self.tooltips_dict,lang)+'\n'+help_texts.get(lang, help_texts["en"])
    
    @classmethod
    def yiu_fit_item_to_mask(self, item_image, item_mask, canvas_image, canvas_mask, object_fit, bg_color):
        b_size, height, width, canvas_channels = canvas_image.shape
        try:
            msk_width, msk_height, msk_left, msk_top = self.mask_bbox(canvas_image, canvas_mask)
        except ValueError as e:
            raise ValueError(f"蒙版边界框计算失败: {str(e)}") from e

        if not bg_color.startswith('#'):
            raise ValueError(f"背景颜色必须以 '#' 开头，当前值: {bg_color}")
        
        bg_color_hex = bg_color.lstrip('#')
        
        if len(bg_color_hex) == 3:
            r = int(bg_color_hex[0] * 2, 16)
            g = int(bg_color_hex[1] * 2, 16)
            b = int(bg_color_hex[2] * 2, 16)
        else:
            r = int(bg_color_hex[0:2], 16)
            g = int(bg_color_hex[2:4], 16)
            b = int(bg_color_hex[4:6], 16)

        batch_size, img_h, img_w, channels = item_image.shape
        image_np = (item_image[0].cpu().numpy() * 255).astype(np.uint8)
        new_width, new_height, x, y = self.fit_or_fill_bbox(msk_width, msk_height, msk_left, msk_top, img_w, img_h, object_fit)

        # 转换 item 为 PIL 图像
        image_np = np.squeeze(image_np)
        pil_item = Image.fromarray(image_np).convert("RGBA")

        resized_item = pil_item.resize((int(new_width), int(new_height)), Image.LANCZOS)

        # 构建透明底图
        transparent_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        transparent_img.paste(resized_item, (int(x), int(y)), mask=resized_item)

        # 正确处理 item 的原始 mask
        item_mask_np = item_mask[0].cpu().numpy()
        if item_mask_np.ndim == 3 and item_mask_np.shape[0] == 1:
            item_mask_np = np.squeeze(item_mask_np, axis=0)
        elif item_mask_np.ndim == 3 and item_mask_np.shape[-1] == 1:
            item_mask_np = np.squeeze(item_mask_np, axis=-1)

        # 转为 PIL mask 并缩放
        pil_item_mask = Image.fromarray((item_mask_np * 255).astype(np.uint8))
        resized_mask = pil_item_mask.resize((int(new_width), int(new_height)), Image.LANCZOS)

        # 粘贴到 canvas 对应位置
        mask_img = Image.new("L", (width, height), 0)
        mask_img.paste(resized_mask, (int(x), int(y)))


        # 背景颜色图像
        bg_color_img = Image.new("RGBA", (width, height), (r, g, b, 255))
        composited_bgcolor = Image.alpha_composite(bg_color_img, transparent_img)

        # 用 canvas_image 作为背景
        canvas_np = (canvas_image[0].cpu().numpy() * 255).astype(np.uint8)
        canvas_pil = Image.fromarray(np.squeeze(canvas_np)).convert("RGBA")
        composited_canvas = Image.alpha_composite(canvas_pil, transparent_img)

        # 转回 tensor
        def pil_to_tensor(pil_img):
            np_img = np.array(pil_img).astype(np.float32) / 255.0
            if np_img.ndim == 2:
                return torch.from_numpy(np_img).unsqueeze(0)  # (1, H, W)
            return torch.from_numpy(np_img).unsqueeze(0)  # (1, H, W, C)

        return (
            pil_to_tensor(transparent_img),                # item_on_transparent
            pil_to_tensor(mask_img),                       # item_mask
            pil_to_tensor(composited_bgcolor),             # item_on_bgcolor
            pil_to_tensor(composited_canvas),              # item_on_canvas
        )
    


    @classmethod
    def mask_bbox(self, image, mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = np.squeeze(mask)
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if image.shape[1:3] != mask.shape:
            raise ValueError(f"尺寸不匹配: 图像尺寸 {image.shape[1:3]} 与蒙版尺寸 {mask.shape} 不符")

        non_zero_indices = np.argwhere(mask > 0)
        if non_zero_indices.size == 0:
            raise ValueError("蒙版内无有效区域，可能未绘制蒙版")

        y_start, x_start = non_zero_indices.min(axis=0).tolist()
        y_end, x_end = non_zero_indices.max(axis=0).tolist()

        width = int(x_end - x_start + 1)
        height = int(y_end - y_start + 1)
        return width, height, int(x_start), int(y_start)

    @staticmethod
    def fit_or_fill_bbox(bw, bh, bx, by, rw, rh, object_fit):
        rect_ratio = rw / rh
        bbox_ratio = bw / bh

        if object_fit == 'fit':
            if rect_ratio > bbox_ratio:
                new_width = bw
                new_height = bw / rect_ratio
            else:
                new_height = bh
                new_width = bh * rect_ratio
        elif object_fit == 'fill':
            if rect_ratio > bbox_ratio:
                new_height = bh
                new_width = bh * rect_ratio
            else:
                new_width = bw
                new_height = bw / rect_ratio
        else:
            raise ValueError("object_fit must be 'fit' or 'fill'")

        new_left = bx + (bw - new_width) / 2
        new_top = by + (bh - new_height) / 2

        return new_width, new_height, new_left, new_top
