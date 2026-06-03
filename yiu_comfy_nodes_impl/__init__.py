from .yiu_img_cnv_placer import yiuImgCnvPlacer
from .yiu_fit_itm_2_msk import yiuFitItm2Msk
from .yiu_itm_cropper import yiuItmCropper
from .yiu_image import yiuImage
from .yiu_save_images import yiuImageSave
from .yiu_img_to_zone_mask import yiuImgToZoneMask
from .yiu_upscale_loop import yiuUpscaleLoopStart, yiuUpscaleLoopEnd
from .internal import (
    yiuWhileLoopStart,
    yiuWhileLoopEnd,
    yiuMathInt,
    yiuCompare,
    yiuTilingMeta,
    yiuImageTile,
    yiuImageUntile,
)

NODE_CLASS_MAPPINGS = {
    "yiuImgCnvPlacer": yiuImgCnvPlacer,
    "yiuFitItm2Msk": yiuFitItm2Msk,
    "yiuItmCropper": yiuItmCropper,
    "yiuImage": yiuImage,
    "yiuImageSave": yiuImageSave,
    "yiuImgToZoneMask": yiuImgToZoneMask,
    "yiuUpscaleLoopStart": yiuUpscaleLoopStart,
    "yiuUpscaleLoopEnd": yiuUpscaleLoopEnd,
    "yiuWhileLoopStart": yiuWhileLoopStart,
    "yiuWhileLoopEnd": yiuWhileLoopEnd,
    "yiuMathInt": yiuMathInt,
    "yiuCompare": yiuCompare,
    "yiuTilingMeta": yiuTilingMeta,
    "yiuImageTile": yiuImageTile,
    "yiuImageUntile": yiuImageUntile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    cls_name: cls.DISPLAY_NAME() if hasattr(cls, "DISPLAY_NAME") else cls_name
    for cls_name, cls in NODE_CLASS_MAPPINGS.items()
}

for cls in NODE_CLASS_MAPPINGS.values():
    if hasattr(cls, "__init_subclass__"):
        cls.__init_subclass__()

print("Loading custom node module: yiu_comfy_nodes_wip")
print(f"NODE_CLASS_MAPPINGS exists: {'NODE_CLASS_MAPPINGS' in globals()}")
print(f"NODE_CLASS_MAPPINGS value: {NODE_CLASS_MAPPINGS}")
print(f"NODE_DISPLAY_NAME_MAPPINGS value: {NODE_DISPLAY_NAME_MAPPINGS}")
