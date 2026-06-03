[中文说明](./README.md) | [English Version](./README_en.md)


# yiu_comfy_nodes

These are small utility tools I frequently use while working with ComfyUI — I hope they’ll be helpful for you too.

## Introduction

The current set of nodes is typically combined for the following use cases:
- Use a product image and a reference environment image to generate e-commerce visuals, with controllable product positioning.
- Combine a product image with a model photo to merge the product into the model’s hand or body, then send it to Kontext for image generation.
- Upscale images to 10,000+ pixels for print materials.

## Node List

- **yiu_fit_itm_2_msk**

    Place Image 1 (item) into the mask area of Image 2 (canvas). Automatically scales the item using object_fit (fit/fill) and centers it into the canvas mask bbox.<br>
    **Inputs:** item_image, item_mask, canvas_image, canvas_mask, object_fit, bg_color (#RRGGBB).<br>
    **Outputs:** item_on_transparent (transparent), item_mask (aligned mask), item_on_bgcolor (composited on bg color), item_on_canvas (composited on original canvas).

- **yiu_img_cnv_placer**

    Places an image (and its mask) onto a canvas of the target resolution with x_pos/y_pos/scale, and generates an expand_mask for outpainting workflows.<br>
    **Inputs:** image, mask, resolution or width_overwrite/height_overwrite, x_pos, y_pos, scale, feather_radius, hex_color.<br>
    **Outputs:** image (transparent canvas), original_mask (aligned), expand_mask (outpaint mask), bg_image (solid background composite), edge_expand_image (edge-radiated background composite).

- **yiu_img_to_zone_mask**

    Generates a rounded-rectangle mask based on the input image size, with controllable position/scale (optional feathering and invert).<br>
    **Inputs:** image (size only), x_pos/y_pos, x_scale/y_scale (optional lock), radius, feather_radius, invert.<br>
    **Outputs:** mask (white inside rect / inverted outside white).

- **yiu_itm_cropper**

    Crops the cutout result based on the cutout mask bbox, and merges the original mask to produce cropped masks.<br>
    **Inputs:** ori_img, ori_mask (optional), cutout_img, cutout_mask, padding, x_pos, y_pos.<br>
    **Outputs:** croped_img (cropped image), croped_msk (cutout_mask × invert(ori_mask)), croped_msk_i (cutout_mask + ori_mask).

- **yiu_upscale_loop**

    Wraps “upscale/super-resolution” into a START/END loop: put any upscaler node between them to upscale in multiple steps to reach a target size, with optional tiling to reduce VRAM spikes.<br>
    **Inputs:** START provides image/target size/per-step scale/tiling params; END receives image_upscaled and advances the loop.<br>
    **Outputs:** END outputs the final full image (loop result).

    - **yiu_upscale_loop_start**

        Loop entry: computes total steps and the scale for this step, tiles internally, then outputs tiles to your upscaler node in the middle.<br>
        **Inputs:** image, target_width/target_height, scale_every_time, max_tile_size, min_tile_overlap_px, tile_overlap_ratio.<br>
        **Outputs:** flow (connect to END), image (tiled batch), scale_this_time, loop_state (connect to END).

    - **yiu_upscale_loop_end**

        Loop exit: receives the upscaled tiles from your upscaler, merges them back to a full image, advances the loop, and outputs the final image when finished.<br>
        **Inputs:** flow (from START), loop_state (from START), image_upscaled (from your upscaler).<br>
        **Outputs:** image (final full image).

## Installation

### Method 1

Search for `yiu_comfy_nodes` in the ComfyUI manager and install it directly.

### Method 2

Download this repository as a ZIP file and put it into your ComfyUI `custom_nodes` folder.

## Usage

Please refer to the demo workflow included in the repository.

## License

This project is licensed under the AGPL license. See the LICENSE file for details.

## Acknowledgments

Thanks to the ComfyUI team and community for their support!
