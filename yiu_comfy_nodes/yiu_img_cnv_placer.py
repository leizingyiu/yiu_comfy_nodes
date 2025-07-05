import torch
import numpy as np
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
                "feather_radius": "羽化半径，扩图时羽化的范围",
            },
            "output":{
                "image": "已经放在扩展画板中的图像",
                "original_mask": "已经放在扩展画板中的原来的遮罩",
                "expand_mask": "扩图节点需要的扩展遮罩",
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
                "feather_radius": "Feather radius, the range of feathering when expanding the image",
            },
            "output":{
                "image": "The image already placed in the extended artboard",
                "original_mask": "The original mask already placed in the extended artboard",
                "expand_mask": "The expanded mask required by the expansion node",
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
                "feather_radius": ("INT", {"default": 20, "min": 0, "max": MAX_RESOLUTION, "step": 1 ,"unit":"px"}),
            }
    
    base_outputs = {
            "image": "IMAGE",
            "original_mask": "MASK",
            "expand_mask": "MASK",
            "help_text": "STRING"
        }
    
  


    MAIN='better_image_pad_for_outpainting'



    @classmethod
    def DISPLAY_NAME(cls):
        return "🖊️ Yiu Image Canvas Placer"
    
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
    def better_image_pad_for_outpainting(self, image, mask=None, resolution="1024x1024", width_overwrite=0, height_overwrite=0, x_pos=50, y_pos=50, scale=100, feather_radius=20):        
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
        new_width,new_height =  map(int, self.scale_rect(img_w, img_h, expanded_width, expanded_height, scale))

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
            if original_mask_tensor is not None:
                original_mask_tensor = original_mask_tensor.repeat(batch_size, 1, 1)
        
        # 返回结果，如果没有原始蒙版则返回全零蒙版
        if original_mask_tensor is not None:
            return (expanded_image_tensor, original_mask_tensor, expand_mask_tensor,)
        else:
            empty_mask = torch.zeros((batch_size, expanded_height, expanded_width))
            return (expanded_image_tensor, empty_mask, expand_mask_tensor,)


    @classmethod
    def scale_rect(self, width, height, container_width, container_height, scale):
        """计算缩放后的矩形尺寸"""
        if scale == 0:
            return (0, 0)
        scale_factor = scale / 100
        if scale == 100:
            scale_factor = min(container_width / width, container_height / height)
        return (width * scale_factor, height * scale_factor)
    
