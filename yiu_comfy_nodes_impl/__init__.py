from .yiu_img_cnv_placer import yiuImgCnvPlacer
from .yiu_fit_itm_2_msk import yiuFitItm2Msk
from .yiu_itm_cropper import yiuItmCropper
from .yiu_img_to_zone_mask import yiuImgToZoneMask
from .yiu_upscale_loop import yiuUpscaleLoopStart, yiuUpscaleLoopEnd
from .internal import (
    INTERNAL_NODE_TYPE_NAMES,
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
    "yiuImgToZoneMask": yiuImgToZoneMask,
    "yiuUpscaleLoopStart": yiuUpscaleLoopStart,
    "yiuUpscaleLoopEnd": yiuUpscaleLoopEnd,
    INTERNAL_NODE_TYPE_NAMES["while_loop_start"]: yiuWhileLoopStart,
    INTERNAL_NODE_TYPE_NAMES["while_loop_end"]: yiuWhileLoopEnd,
    INTERNAL_NODE_TYPE_NAMES["math_int"]: yiuMathInt,
    INTERNAL_NODE_TYPE_NAMES["compare"]: yiuCompare,
    INTERNAL_NODE_TYPE_NAMES["tiling_meta"]: yiuTilingMeta,
    INTERNAL_NODE_TYPE_NAMES["image_tile"]: yiuImageTile,
    INTERNAL_NODE_TYPE_NAMES["image_untile"]: yiuImageUntile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    cls_name: cls.DISPLAY_NAME() if hasattr(cls, "DISPLAY_NAME") else cls_name
    for cls_name, cls in NODE_CLASS_MAPPINGS.items()
}

NODE_DISPLAY_NAME_MAPPINGS.update(
    {
        INTERNAL_NODE_TYPE_NAMES["while_loop_start"]: ".internal/while_loop_start",
        INTERNAL_NODE_TYPE_NAMES["while_loop_end"]: ".internal/while_loop_end",
        INTERNAL_NODE_TYPE_NAMES["math_int"]: ".internal/math_int",
        INTERNAL_NODE_TYPE_NAMES["compare"]: ".internal/compare",
        INTERNAL_NODE_TYPE_NAMES["tiling_meta"]: ".internal/tiling_meta",
        INTERNAL_NODE_TYPE_NAMES["image_tile"]: ".internal/image_tile",
        INTERNAL_NODE_TYPE_NAMES["image_untile"]: ".internal/image_untile",
    }
)

for cls in NODE_CLASS_MAPPINGS.values():
    if hasattr(cls, "__init_subclass__"):
        cls.__init_subclass__()

print("Loading custom node module: yiu_comfy_nodes_wip")
print(f"NODE_CLASS_MAPPINGS exists: {'NODE_CLASS_MAPPINGS' in globals()}")
print(f"NODE_CLASS_MAPPINGS value: {NODE_CLASS_MAPPINGS}")
print(f"NODE_DISPLAY_NAME_MAPPINGS value: {NODE_DISPLAY_NAME_MAPPINGS}")
