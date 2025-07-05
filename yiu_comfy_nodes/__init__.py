from .yiu_img_cnv_placer import yiuImgCnvPlacer
from .yiu_fit_itm_2_msk import yiuFitItm2Msk
from .yiu_itm_cropper import yiuItmCropper

NODE_CLASS_MAPPINGS = {
    "yiuImgCnvPlacer": yiuImgCnvPlacer,
    "yiuFitItm2Msk": yiuFitItm2Msk,
    "yiuItmCropper": yiuItmCropper
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