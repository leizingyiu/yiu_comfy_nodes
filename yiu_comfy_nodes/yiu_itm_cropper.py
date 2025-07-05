import torch
from PIL import Image, ImageChops, ImageOps
import numpy as np
from .utils import get_current_language,get_localized_tooltips
from .yiu_base_node import YiuBaseNode

print('this is yiu_itm_cropper.py - 202507051718')

class yiuItmCropper(YiuBaseNode):
 
    tooltips_dict={
        "zh-CN":{
            "input": {
                "ori_img": "原始图像",
                "ori_mask": "原始遮罩",
                "cutout_img": "抠图结果图像",
                "cutout_mask": "抠图结果遮罩",
                "padding": "四周保留的边距",
                "x_pos":"在边距中，图片在左边还是右边",
                "y_pos": "在边距中，图片在上边还是下边",
            },
            "output":{
                "croped img":"裁切后的图像",
                "croped msk":"裁切并合并(减)的遮罩",
                "croped_msk_i":"裁切并合并(加)的遮罩"
            }
        },
        "en":{
            "input": {
                "ori_img": "Original image",
                "ori_mask": "Original mask",
                "cutout_img": "Cutout result image",
                "cutout_mask": "Cutout result mask",
                "padding": "The padding reserved around",
                "x_pos": "In the padding, is the image on the left or right",
                "y_pos": "In the padding, is the image on the top or bottom",
            },
            "output":{
                "croped img":"Cropped image",
                "croped msk":"cropped and merged (subtracted) mask",
                "croped_msk_i":"cropped and merged (added) mask"
            }
        }
    }

    base_inputs = {
        "ori_img": ("IMAGE", {}),
        "ori_mask": ("MASK", {"default": None}),
        "cutout_img": ("IMAGE", {}),
        "cutout_mask": ("MASK", {}),
        "padding": ("INT", {"default": 0, "min": 0, "max": 500 ,"unit":"px"}),
        "x_pos": ("INT", {"default": 0, "min": -100, "max": 100 ,"unit":"px"}),
        "y_pos": ("INT", {"default": 0, "min": -100, "max": 100 ,"unit":"px" }),
    }

    base_outputs = {
        "croped_img": "IMAGE",
        "croped_msk": "MASK",
        "croped_msk_i": "MASK",
        "help_text": "STRING"
    }

   


    MAIN = "yiu_itm_cropper"

    @classmethod
    def DISPLAY_NAME(cls):
        return "✂ Yiu Item Cropper"
    
    @classmethod
    def get_help_text(self):
        lang = get_current_language()
        help_texts = {
            "zh-CN": "📖 使用说明：输入原图的图像和遮罩，以及抠图结果的图像和遮罩，根据抠图结果遮罩的大小和位置，对抠图结果进行裁切输出 croped_img ; 将原图遮罩反转叠再抠图结果遮罩上，并裁切输出为 croped_msk。",
            "en": "📖 Instructions for use: Input the image and mask of the original image, as well as the image and mask of the cutout result. According to the size and position of the cutout result mask, crop the cutout result and output it as croped_img ; invert the original image mask and overlay it on the cutout result mask, and crop and output it as croped_msk ."
        }
        return  get_localized_tooltips(self.tooltips_dict,lang)+'\n'+ help_texts.get(lang, help_texts["en"])
    



    @classmethod
    def yiu_itm_cropper(cls, ori_img, ori_mask=None, cutout_img=None, cutout_mask=None,
                        padding=0, x_pos=0, y_pos=0):
        if cutout_img is None:
            raise ValueError("Error: 'cutout_img' is a required input but was not provided.")
        if cutout_mask is None:
            raise ValueError("Error: 'cutout_mask' is a required input but was not provided.")

        ori_img_np = (ori_img.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        cutout_img_np = (cutout_img.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        cutout_mask_np = (cutout_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

        pil_ori_img = Image.fromarray(ori_img_np)
        pil_cutout_img = Image.fromarray(cutout_img_np)
        pil_cutout_mask = Image.fromarray(cutout_mask_np).convert("L")

        if ori_mask is None:
            pil_ori_mask = Image.new("L", pil_ori_img.size, 255)
        else:
            pil_ori_mask = Image.fromarray((ori_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)).convert("L")

        bbox = pil_cutout_mask.getbbox()

        if bbox is None:
            black_mask_pil = Image.new("L", pil_ori_img.size, 0)
            output_cropped_img = torch.from_numpy(np.array(pil_cutout_img).astype(np.float32) / 255.0).unsqueeze(0)
            output_processed_mask = torch.from_numpy(np.array(black_mask_pil).astype(np.float32) / 255.0).unsqueeze(0)
            return (output_cropped_img, output_processed_mask, output_processed_mask, "cutout_mask 全黑，未找到边界框。")

        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1

        left_pad   = int(padding * (1 + x_pos / 100))
        right_pad  = int(padding * (1 - x_pos / 100))
        top_pad    = int(padding * (1 + y_pos / 100))
        bottom_pad = int(padding * (1 - y_pos / 100))

        x1_new = max(0, x1 - left_pad)
        y1_new = max(0, y1 - top_pad)
        x2_new = min(pil_cutout_img.width,  x2 + right_pad)
        y2_new = min(pil_cutout_img.height, y2 + bottom_pad)

        cropped_img = pil_cutout_img.crop((x1_new, y1_new, x2_new, y2_new))
        _msk = pil_ori_mask.crop((x1_new, y1_new, x2_new, y2_new))
        msk  = pil_cutout_mask.crop((x1_new, y1_new, x2_new, y2_new))

        _msk_inverted = ImageOps.invert(_msk)
        blended_mask_pil = ImageChops.multiply(msk, _msk_inverted)

        blended_mask_i_pil = ImageChops.add(msk, _msk)

        processed_mask_np = np.array(blended_mask_pil).astype(np.float32) / 255.0
        processed_mask = torch.from_numpy(processed_mask_np).unsqueeze(0)

        processed_mask_i_np = np.array(blended_mask_i_pil).astype(np.float32) / 255.0
        processed_mask_i = torch.from_numpy(processed_mask_i_np).unsqueeze(0)

        cropped_img_np = np.array(cropped_img).astype(np.float32) / 255.0
        cropped_img_tensor = torch.from_numpy(cropped_img_np).unsqueeze(0)

        help_text = f"裁剪完成。原bbox: ({x1},{y1},{x2},{y2}) → 新bbox: ({x1_new},{y1_new},{x2_new},{y2_new})"

        return (cropped_img_tensor, processed_mask, processed_mask_i)

  