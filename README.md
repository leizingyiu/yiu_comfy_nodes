# yiu_comfy_nodes

[中文说明](./README.md) | [English Version](./README_en.md)

这是我在用comfyUI的过程中，频繁用到的小工具，希望对你也有用。

## 介绍

当前几个节点组合起来最常见的引用场景：
- 使用产品图以及环境结构参考图，在生成电商图片时，更可控的调整产品位置
- 使用产品图，以及模特图，合并产品到模特手上或身上，喂给kontext生图
- 将图片放大到10000以上，为印刷物料撑图

## 节点列表

- **yiu_fit_itm_2_msk**

    将图1(item)放置到图2(canvas)的遮罩区域中，自动按 object_fit（fit/fill）缩放并居中到遮罩 bbox。<BR>
    输入：item_image、item_mask，canvas_image、canvas_mask，object_fit，bg_color（#RRGGBB）<BR>
    输出：item_on_transparent（透明底）、item_mask（对齐后的遮罩）、item_on_bgcolor（背景色合成）、item_on_canvas（原背景合成）


- **yiu_img_cnv_placer**

    将图片（及其遮罩）按 x_pos/y_pos/scale 放进指定分辨率画布，生成可用于扩图(outpaint)的 expand_mask。<BR>
    输入：image、mask，resolution 或 width/height_overwrite，x_pos、y_pos、scale、feather_radius、hex_color<BR>
    输出：image（透明底画布）、original_mask（对齐后的原遮罩）、expand_mask（扩图遮罩）、bg_image（纯色背景合成）、edge_expand_image（边缘放射扩展背景合成）

- **yiu_img_to_zone_mask**

    根据输入 image 的尺寸生成一个可控位置/大小的圆角矩形遮罩（可选羽化与反转）。<BR>
    输入：image（仅读取尺寸），x_pos/y_pos，x_scale/y_scale（可锁定），radius，feather_radius，invert<BR>
    输出：mask（矩形内白/反转外白）

- **yiu_itm_cropper**

    根据抠图结果遮罩的 bbox 对抠图结果裁切，并把原图遮罩参与合并生成裁切后的遮罩。<BR>
    输入：ori_img、ori_mask（可选），cutout_img、cutout_mask，padding、x_pos、y_pos<BR>
    输出：croped_img（裁切图）、croped_msk（cutout_mask × invert(ori_mask)）、croped_msk_i（cutout_mask + ori_mask）

- **yiu_upscale_loop**

    将“放大/超分”封装为 START/END 两个节点的循环结构：把任意放大节点夹在中间，可分多步放大到目标尺寸，并可选 tiling 拼接避免单次爆显存。<BR>
    输入：START 提供 image/目标尺寸/每步倍率/tiling 参数；END 接收放大后的 image_upscaled 并推进循环<BR>
    输出：END 输出最终整图 image（循环结束结果）

    - **yiu_upscale_loop_start**

        放大循环起点：根据目标尺寸计算总步数与本步倍率，并在内部做 tile 后把 tiles 输出给中间放大节点。<BR>
        输入：image，target_width/target_height，scale_every_time，max_tile_size，min_tile_overlap_px，tile_overlap_ratio<BR>
        输出：flow（连到 END）、image（tiles 批量）、scale_this_time、loop_state（连到 END）

    - **yiu_upscale_loop_end**

        放大循环终点：接收中间放大节点输出的 tiles，拼回整图并驱动下一步循环，结束时输出最终图像。<BR>
        输入：flow（来自 START）、loop_state（来自 START）、image_upscaled（中间放大节点输出）<BR>
        输出：image（最终整图）



## 安装方法

### 方法 1 

直接在 comfyUI 的 manager 中搜索 yiu_comfy_nodes 安装

### 方法 2

下载本仓库zip文件， 放到 comfyUI 的 custom_nodes 文件夹中。


## 使用说明

请参考仓库中的 DEMO 工作流

## 许可协议

本项目遵循 AGPL 协议，详情请参阅 LICENSE 文件。

## 致谢

感谢 ComfyUI 团队及社区的支持！
