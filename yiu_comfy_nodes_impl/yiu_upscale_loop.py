import json
import math
import urllib.request
from .utils import (
    get_current_language,
    get_localized_tooltips,
    derive_target_size,
    compute_step_total,
    compute_scale_this_time,
    compute_tiling_params,
)
from .yiu_base_node import YiuBaseNode
from .internal import yiuImageTile, yiuImageUntile

try:
    from comfy_execution.graph_utils import GraphBuilder
except Exception:
    GraphBuilder = None

try:
    from server import PromptServer
except Exception:
    PromptServer = None

# try:
#     from nodes import MAX_RESOLUTION
# except Exception:
#     MAX_RESOLUTION = 8192

MAX_RESOLUTION=9999999
DEBUG__UPSCALE_LOOP = False


def _debug_print(msg):
    if DEBUG__UPSCALE_LOOP:
        print(msg)


_debug_print("this is yiu_upscale_loop.py - 202605310002")


def _report_debug_event(hypothesis_id, location, msg, data):
    # #region debug-point shared:report-event
    try:
        server_url = "http://127.0.0.1:7777/event"
        session_id = "upscale-loop-regress"
        try:
            with open(".dbg/upscale-loop-regress.env", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("DEBUG_SERVER_URL="):
                        server_url = line.split("=", 1)[1].strip() or server_url
                    elif line.startswith("DEBUG_SESSION_ID="):
                        session_id = line.split("=", 1)[1].strip() or session_id
        except Exception:
            pass
        payload = {
            "sessionId": session_id,
            "runId": "post-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data,
        }
        urllib.request.urlopen(
            urllib.request.Request(
                server_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.2,
        ).read()
    except Exception:
        pass
    # #endregion


class yiuUpscaleLoopStart(YiuBaseNode):
    tooltips_dict = {
        "zh-CN": {
            "input": {
                "image": "输入图像（第 0 步的初始图像）",
                "target_width": "目标宽度（px）。为 0 时会根据 target_height 按原图比例推导",
                "target_height": "目标高度（px）。为 0 时会根据 target_width 按原图比例推导",
                "scale_every_time": "每一步的基础放大倍率（>1）。最后一步会自动补足到目标倍率",
                "max_tile_size": "tile 最大尺寸（px）。建议 1024/1536。设为 0 表示关闭 tiling（整图直通）",
                "min_tile_overlap_px": "最小接缝宽度（px），对应 default_OVERLAP 这种固定值。设为 0 表示接缝最小值为 0",
                "tile_overlap_ratio": "接缝比例（0~1）。与 min_tile_overlap_px 组合更稳",
            },
            "output": {
                "flow": "循环连线（必须连接到 END.flow）",
                "image": "本步要交给用户放大节点处理的图像（循环内会自动更新为上一步输出）",
                "scale_this_time": "本步推荐放大倍率（大多数步=scale_every_time，最后一步=补足倍率）",
                "loop_state": "内部循环状态，连接到 END.loop_state",
                "print(step_index)": "输出当前步索引（从 0 开始）",
                "print(step_total)": "输出总步数",
                "image_last_step": "本轮开始时的整图，也就是上一轮结束后的整图，可用于备份/保存",
                "help_text": "帮助信息",
            },
        },
        "en": {
            "input": {
                "image": "Input image (initial image for step 0)",
                "target_width": "Target width (px). If 0, derived from target_height with original aspect ratio",
                "target_height": "Target height (px). If 0, derived from target_width with original aspect ratio",
                "scale_every_time": "Base upscale factor per step (>1). The last step auto-fills to reach target",
                "max_tile_size": "Max tile size (px). Suggested 1024/1536. Set 0 to disable tiling",
                "min_tile_overlap_px": "Minimum seam width in pixels (fixed overlap like default_OVERLAP). Set 0 to remove the minimum",
                "tile_overlap_ratio": "Seam ratio (0~1). Recommended to combine with min_tile_overlap_px",
            },
            "output": {
                "flow": "Loop link (must connect to END.flow)",
                "image": "Image for the user upscaler (auto-updated in the loop)",
                "scale_this_time": "Recommended scale for this step (mostly scale_every_time, last step auto-fill)",
                "loop_state": "Internal loop state, connect to END.loop_state",
                "print(step_index)": "Outputs current step index (0-based)",
                "print(step_total)": "Outputs total steps",
                "image_last_step": "Full image at the start of this round, i.e. the previous round result",
                "help_text": "Help text",
            },
        },
    }

    base_inputs = {
        "image": ("IMAGE", {}),
        "target_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "unit": "px"}),
        "target_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "unit": "px"}),
        "scale_every_time": ("FLOAT", {"default": 2.0, "min": 1.001, "max": 16.0, "step": 0.001}),
        "max_tile_size": ("INT", {"default": 1024, "min": 48, "max": MAX_RESOLUTION, "unit": "px"}),
        "min_tile_overlap_px": ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "unit": "px"}),
        "tile_overlap_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
    }

    base_optional_inputs = {
    }

    base_hidden_inputs = {
        "_loop_index_in": "INT",
        "_loop_image_in": "IMAGE",
        # "_loop_index_in": ("INT", {"forceInput": True}),
        # "_loop_image_in": ("IMAGE", {"forceInput": True}),
        "prompt": "PROMPT",
        "extra_pnginfo": "EXTRA_PNGINFO",
        "unique_id": "UNIQUE_ID",
    }

    base_outputs = {
        "yiu_upscale_loop_flow": "FLOW_CONTROL",
        "image": "IMAGE",
        "scale_this_time": "FLOAT",
        "loop_state": "YIU_UPSCALE_LOOP_STATE",
        "print(step_index)": "INT",
        "print(step_total)": "INT",
        "image_last_step": "IMAGE",
        "help_text": "STRING",
    }

    MAIN = "upscale_loop_start"

    @classmethod
    def DISPLAY_NAME(cls):
        return "🔁 Upscale Loop START"

    @classmethod
    def get_help_text(self):
        lang = get_current_language()
        help_texts = {
            "zh-CN": "📖 这是循环的起点。把任意放大节点夹在 START 和 END 之间，并连接 START.flow -> END.flow。",
            "en": "Loop entry. Put any upscaler between START and END, and connect START.flow -> END.flow.",
        }
        return get_localized_tooltips(self.tooltips_dict, lang) + "\n" + help_texts.get(lang, help_texts["en"])

    @classmethod
    def upscale_loop_start(
        cls,
        image,
        target_width,
        target_height,
        scale_every_time,
        max_tile_size,
        min_tile_overlap_px,
        tile_overlap_ratio,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        _loop_index_in=None,
        _loop_image_in=None,
    ):
        if GraphBuilder is None:
            raise RuntimeError("GraphBuilder is not available. This node must run inside ComfyUI.")

        _, h0, w0, _ = image.shape
        target_w, target_h = derive_target_size(w0, h0, target_width, target_height)
        total_scale, step_total = compute_step_total(w0, h0, target_w, target_h, scale_every_time)

        current_index = int(_loop_index_in) if _loop_index_in is not None else 0
        current_image = _loop_image_in if _loop_image_in is not None else image

        # #region debug-point B:start-inputs
        _report_debug_event(
            "B",
            "yiu_upscale_loop.py:start",
            "[DEBUG] START received loop-carried inputs",
            {
                "unique_id": str(unique_id),
                "loop_index_in": None if _loop_index_in is None else str(_loop_index_in),
                "loop_image_in_is_none": _loop_image_in is None,
                "image_shape": tuple(int(x) for x in image.shape),
                "current_image_shape": tuple(int(x) for x in current_image.shape),
                "target_width": int(target_width),
                "target_height": int(target_height),
                "scale_every_time": float(scale_every_time),
            },
        )
        # #endregion

        _, ih, iw, _ = current_image.shape
        rows, cols, overlap_x, overlap_y = compute_tiling_params(
            int(ih), int(iw), int(max_tile_size), int(min_tile_overlap_px), float(tile_overlap_ratio)
        )

        if current_index < 0:
            current_index = 0
        if current_index >= int(step_total):
            current_index = int(step_total) - 1

        scale_this_time = compute_scale_this_time(total_scale, scale_every_time, step_total, current_index)

        # #region debug-point C:start-state
        _report_debug_event(
            "C",
            "yiu_upscale_loop.py:start",
            "[DEBUG] START computed loop state",
            {
                "unique_id": str(unique_id),
                "current_index": int(current_index),
                "step_total": int(step_total),
                "total_scale": float(total_scale),
                "scale_this_time": float(scale_this_time),
                "rows": int(rows),
                "cols": int(cols),
                "overlap_x": int(overlap_x),
                "overlap_y": int(overlap_y),
            },
        )
        # #endregion

        s = float(scale_every_time)
        log_steps = None
        if math.isfinite(float(total_scale)) and float(total_scale) > 0 and math.isfinite(s) and s > 1.0:
            try:
                log_steps = float(math.log(float(total_scale)) / math.log(s))
            except Exception:
                log_steps = None

        total_scale_floor = int(math.floor(float(total_scale))) if math.isfinite(float(total_scale)) else None
        total_scale_frac = float(total_scale) - float(total_scale_floor) if total_scale_floor is not None else None
        last_step_scale = None
        if int(step_total) > 1 and math.isfinite(float(total_scale)) and math.isfinite(s) and s > 1.0:
            try:
                last_step_scale = float(float(total_scale) / (s ** max(0, int(step_total) - 1)))
            except Exception:
                last_step_scale = None

        loop_state = {
            "scale": float(scale_this_time),
            "step_index": int(current_index),
            "step_total": int(step_total),
            "orig_h": int(ih),
            "orig_w": int(iw),
            "rows": int(rows),
            "cols": int(cols),
            "overlap_x": int(overlap_x),
            "overlap_y": int(overlap_y),
            "lap_x": int(overlap_x),
            "lap_y": int(overlap_y),
            "scale_every_time": float(scale_every_time),
            "total_scale": float(total_scale),
            "log_steps": None if log_steps is None else float(log_steps),
            "total_scale_floor": None if total_scale_floor is None else int(total_scale_floor),
            "total_scale_frac": None if total_scale_frac is None else float(total_scale_frac),
            "last_step_scale": None if last_step_scale is None else float(last_step_scale),
            "target_w": int(target_w),
            "target_h": int(target_h),
            "w0": int(w0),
            "h0": int(h0),
            "current_image_h": int(ih),
            "current_image_w": int(iw),
            "max_tile_size": int(max_tile_size),
            "min_tile_overlap_px": int(min_tile_overlap_px),
            "tile_overlap_ratio": float(tile_overlap_ratio),
        }

        _debug_print(
            "[yiu_upscale_loop] START loop_state="
            + json.dumps(loop_state, ensure_ascii=False, sort_keys=True)
            + f" unique_id={str(unique_id)}"
        )

        graph = GraphBuilder()

        tiled_image = yiuImageTile.tile(
            current_image,
            loop_state["orig_h"],
            loop_state["orig_w"],
            loop_state["rows"],
            loop_state["cols"],
            loop_state["overlap_x"],
            loop_state["overlap_y"],
        )[0]

        # #region debug-point C:start-return
        _report_debug_event(
            "C",
            "yiu_upscale_loop.py:start:return",
            "[DEBUG] START prepared outputs",
            {
                "unique_id": str(unique_id),
                "current_index": int(current_index),
                "step_total": int(step_total),
                "scale_this_time": float(scale_this_time),
                "loop_state_step_index": int(loop_state["step_index"]),
                "loop_state_step_total": int(loop_state["step_total"]),
                "tiled_image_shape": tuple(int(x) for x in tiled_image.shape),
            },
        )
        # #endregion

        return {
            "result": (
                "stub",
                tiled_image,
                float(scale_this_time),
                loop_state,
                int(current_index),
                int(step_total),
                current_image,
            ),
            "expand": graph.finalize(),
        }


class yiuUpscaleLoopEnd(YiuBaseNode):
    tooltips_dict = {
        "zh-CN": {
            "input": {
                "flow": "来自 START.flow 的循环连线（rawLink）",
                "loop_state": "来自 START.loop_state 的内部循环状态",
                "image_upscaled": "来自用户放大节点（或节点组）的输出图像（会作为下一步输入图像）",
            },
            "output": {
                "image": "循环结束后的最终图像",
                "image_step": "本步产出的图像（用于循环体内外接保存节点）",
                "print(last_scale_applied)": "输出本步倍率（来自 START.scale_this_time）",
                "print(step_index)": "输出当前步索引（来自 START）",
                "print(step_total)": "输出总步数（来自 START）",
                "help_text": "帮助信息",
            },
        },
        "en": {
            "input": {
                "flow": "Loop link from START.flow (rawLink)",
                "loop_state": "Internal loop state from START.loop_state",
                "image_upscaled": "Output from the user upscaler (used as the next-step input image)",
            },
            "output": {
                "image": "Final image after the loop ends",
                "image_step": "Per-step image (connect to a save node inside the loop if desired)",
                "print(last_scale_applied)": "Scale used in this step (from START.scale_this_time)",
                "print(step_index)": "Current step index (from START)",
                "print(step_total)": "Total steps (from START)",
                "help_text": "Help text",
            },
        },
    }

    base_inputs = {
        "yiu_upscale_loop_flow": ("FLOW_CONTROL", {"rawLink": True}),
        "loop_state": ("YIU_UPSCALE_LOOP_STATE", {}),
        "image_upscaled": ("IMAGE", {}),
    }

    base_hidden_inputs = {
        "dynprompt": "DYNPROMPT",
        "extra_pnginfo": "EXTRA_PNGINFO",
        "unique_id": "UNIQUE_ID",
    }

    base_outputs = {
        "image": "IMAGE",
        # "image_step": "IMAGE",
        # "print(last_scale_applied)": "FLOAT",
        # "print(step_index)": "INT",
        # "print(step_total)": "INT",
        "help_text": "STRING",
    }

    MAIN = "upscale_loop_end"

    @classmethod
    def DISPLAY_NAME(cls):
        return "🔁 Upscale Loop END"

    @classmethod
    def get_help_text(self):
        lang = get_current_language()
        help_texts = {
            "zh-CN": "📖 这是循环的终点。接收放大后的 image_upscaled，推进循环索引，并把图像回传给下一步。",
            "en": "Loop exit. Receives image_upscaled, advances the loop index, and feeds the image into the next step.",
        }
        return get_localized_tooltips(self.tooltips_dict, lang) + "\n" + help_texts.get(lang, help_texts["en"])

    @classmethod
    def upscale_loop_end(cls, yiu_upscale_loop_flow, loop_state, image_upscaled, dynprompt=None, extra_pnginfo=None, unique_id=None):
        flow = yiu_upscale_loop_flow
        if GraphBuilder is None:
            raise RuntimeError("GraphBuilder is not available. This node must run inside ComfyUI.")

        start_scale = float(loop_state["scale"])
        start_index = int(loop_state["step_index"])
        step_total = int(loop_state["step_total"])
        next_index = start_index + 1
        should_continue = next_index < step_total

        recurse_target = None
        try:
            if dynprompt is not None:
                recurse_target = dynprompt.get_display_node_id(str(unique_id))
        except Exception:
            recurse_target = None
        if recurse_target is None:
            recurse_target = str(unique_id).split(".")[0]

        _debug_print(
            "[yiu_upscale_loop] END loop_state="
            + json.dumps(loop_state, ensure_ascii=False, sort_keys=True)
            + f" should_continue={bool(should_continue)} next_index={int(next_index)} flow={str(flow)} unique_id={str(unique_id)} recurse_target={str(recurse_target)}"
        )

        # #region debug-point D:end-state
        _report_debug_event(
            "D",
            "yiu_upscale_loop.py:end",
            "[DEBUG] END prepared next loop state",
            {
                "unique_id": str(unique_id),
                "flow": str(flow),
                "start_index": int(start_index),
                "next_index": int(next_index),
                "step_total": int(step_total),
                "should_continue": bool(should_continue),
                "image_upscaled_shape": tuple(int(x) for x in image_upscaled.shape),
            },
        )
        # #endregion

        merged = yiuImageUntile.untile(
            image_upscaled,
            loop_state["scale"],
            loop_state["orig_h"],
            loop_state["orig_w"],
            loop_state["rows"],
            loop_state["cols"],
            loop_state["overlap_x"],
            loop_state["overlap_y"],
        )[0]

        if not should_continue and PromptServer is not None:
            try:
                _, merged_h, merged_w, _ = merged.shape
                merged_h = int(merged_h)
                merged_w = int(merged_w)
                merged_pixels = merged_h * merged_w
                if merged_pixels > 200_000_000:
                    server = getattr(PromptServer, "instance", None)
                    if server is not None:
                        server.send_sync(
                            "yiu_upscale_loop_preview_warning",
                            {
                                "node": str(recurse_target),
                                "width": merged_w,
                                "height": merged_h,
                                "pixels": int(merged_pixels),
                                "threshold": 200_000_000,
                                "message_zh": (
                                    f"图像输出为 {merged_w}x{merged_h}（{merged_pixels} 像素），超过 200000000 像素。"
                                    "ComfyUI 的预览节点可能不显示。请连接保存节点保存，或到 ComfyUI 的 temp 文件夹查看生成的图片。"
                                ),
                                "message_en": (
                                    f"Output image is {merged_w}x{merged_h} ({merged_pixels} pixels), above 200000000 pixels. "
                                    "ComfyUI preview nodes may fail to display it. Use a Save node, or check ComfyUI's temp folder."
                                ),
                            },
                        )
            except Exception:
                pass

        graph = GraphBuilder()
        while_close = graph.node(
            "yiuWhileLoopEnd",
            flow=flow,
            condition=bool(should_continue),
            recurse_target=str(recurse_target),
            initial_value0=int(next_index),
            initial_value1=merged,
            initial_value2=float(start_scale),
            initial_value3=int(start_index),
            initial_value4=int(step_total),
        )

        # #region debug-point D:end-while-seed
        _report_debug_event(
            "D",
            "yiu_upscale_loop.py:end:while-seed",
            "[DEBUG] END seeded recursive outputs",
            {
                "unique_id": str(unique_id),
                "should_continue": bool(should_continue),
                "initial_value0_next_index": int(next_index),
                "initial_value2_start_scale": float(start_scale),
                "initial_value3_start_index": int(start_index),
                "initial_value4_step_total": int(step_total),
            },
        )
        # #endregion

        image_out = while_close.out(0) if should_continue else merged
        image_step_out = while_close.out(1) if should_continue else merged
        scale_out = while_close.out(2) if should_continue else float(start_scale)
        step_index_out = while_close.out(3) if should_continue else int(start_index)
        step_total_out = while_close.out(4) if should_continue else int(step_total)

        # #region debug-point E:end-outputs
        _report_debug_event(
            "E",
            "yiu_upscale_loop.py:end",
            "[DEBUG] END selected outputs",
            {
                "unique_id": str(unique_id),
                "should_continue": bool(should_continue),
                "scale_out_type": type(scale_out).__name__,
                "step_index_out_type": type(step_index_out).__name__,
                "step_total_out_type": type(step_total_out).__name__,
            },
        )
        # #endregion

        return {
            "result": (
                image_out,
                # image_step_out,
                # scale_out,
                # step_index_out,
                # step_total_out,
            ),
            "expand": graph.finalize(),
        }
