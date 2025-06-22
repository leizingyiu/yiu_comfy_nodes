# __init__.py

# Import the basic node class
from .yiu_img_cnv_placer import yiuImgCnvPlacer
from .yiu_fit_itm_2_msk import yiuFitItm2Msk

# Register the nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "YiuImgCnvPlacer": yiuImgCnvPlacer,
    "yiuFitItm2Msk": yiuFitItm2Msk,
}  # 确保没有隐藏字符干扰

# 自动从节点类中获取显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    cls_name: cls.DISPLAY_NAME() if hasattr(cls, "DISPLAY_NAME") else cls_name
    for cls_name, cls in NODE_CLASS_MAPPINGS.items()
}

# Debug statement to verify module loading
print("Loading custom node module: yiu_comfy_nodes_wip")
print(f"NODE_CLASS_MAPPINGS exists: {'NODE_CLASS_MAPPINGS' in globals()}")
print(f"NODE_CLASS_MAPPINGS value: {NODE_CLASS_MAPPINGS}")
print(f"NODE_DISPLAY_NAME_MAPPINGS value: {NODE_DISPLAY_NAME_MAPPINGS}")