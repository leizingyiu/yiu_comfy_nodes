
# yiu_comfy_nodes

These are small utility tools I frequently use while working with ComfyUI — I hope they’ll be helpful for you too.

## Introduction

The current set of nodes is typically combined for the following use cases:
- Use a product image and a reference environment image to generate e-commerce visuals, with controllable product positioning.
- Combine a product image with a model photo to merge the product into the model’s hand or body, then send it to Kontext for image generation.

## Node List

- **yiu_fit_itm_2_msk**

    Place an item from Image 1 into the mask area of Image 2.<br>
    **Inputs:** item image, mask image, background image, and a mask on the background indicating the item’s position and size (can be manually drawn in the Load Image node).<br>
    **Outputs:** item image (with transparent background) scaled and positioned in the mask, item’s mask, item image with background color, and item image with original background.

- **yiu_img_cnv_placer**

    Places an image into a canvas with a specified resolution, allowing customization of the image’s scaling and position within the canvas.<br>
    **Inputs:** image and its mask.<br>
    **Outputs:** image placed onto the canvas, original mask placed onto the canvas, and a mask for upscaling.

- **yiu_itm_cropper**

    Crops an image and its mask according to a cutout result.<br>
    **Inputs:** original image, original mask, cutout result image, and cutout result mask.<br>
    **Outputs:** cropped image and mask based on the cutout result and adjustments via padding, x_pos, and y_pos.

## Installation

### Method 1

Search for `yiu_comfy_nodes` in the ComfyUI manager and install it directly.

### Method 2

Download the `yiu_comfy_nodes` folder from this repository and copy it into your ComfyUI `custom_nodes` folder.

## Usage

Please refer to the demo workflow included in the repository:
![yiu_comfy_nodes_demo](https://raw.githubusercontent.com/leizingyiu/yiu_comfy_nodes/refs/heads/main/yiu_demo_workflow.png "This is a demo workflow image")

## License

This project is licensed under the AGPL license. See the LICENSE file for details.

## Acknowledgments

Thanks to the ComfyUI team and community for their support!
