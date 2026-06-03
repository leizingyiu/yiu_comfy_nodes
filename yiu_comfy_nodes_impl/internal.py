import json
import math
import torch
import urllib.request
from .utils import (
    get_current_language,
    get_localized_tooltips,
    yiu_any_type,
    yiu_bypass_type_tuple,
    yiu_max_flow_num,
    yiu_scalar,
    yiu_int,
    yiu_float,
    compute_tiling_params,
    tile_image_batch,
    untile_image_batch,
)
from .yiu_base_node import YiuBaseNode

try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
except Exception:
    GraphBuilder = None
    is_link = None

try:
    from comfy_execution.graph import ExecutionBlocker
except Exception:
    ExecutionBlocker = None

try:
    from nodes import MAX_RESOLUTION, NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
except Exception:
    MAX_RESOLUTION = 8192
    ALL_NODE_CLASS_MAPPINGS = {}

DEBUG__INTERNAL_LOOP = False


def _debug_print(msg):
    if DEBUG__INTERNAL_LOOP:
        print(msg)


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


class yiuWhileLoopStart:
    CATEGORY = "yiu_nodes/_internal"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required": {"condition": ("BOOLEAN", {"default": True})}, "optional": {}}
        for i in range(yiu_max_flow_num()):
            inputs["optional"][f"initial_value{i}"] = (yiu_any_type(),)
        return inputs

    RETURN_TYPES = yiu_bypass_type_tuple(tuple(["FLOW_CONTROL"] + [yiu_any_type()] * yiu_max_flow_num()))
    RETURN_NAMES = yiu_bypass_type_tuple(tuple(["flow"] + [f"value{i}" for i in range(yiu_max_flow_num())]))
    FUNCTION = "while_loop_open"

    def while_loop_open(self, condition, **kwargs):
        values = []
        for i in range(yiu_max_flow_num()):
            if condition:
                values.append(kwargs.get(f"initial_value{i}", None))
            else:
                values.append(ExecutionBlocker(None) if ExecutionBlocker is not None else None)
        return tuple(["stub"] + values)


class yiuWhileLoopEnd:
    CATEGORY = "yiu_nodes/_internal"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "condition": ("BOOLEAN", {}),
            },
            "optional": {
                "recurse_target": ("STRING", {"default": ""}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
        for i in range(yiu_max_flow_num()):
            inputs["optional"][f"initial_value{i}"] = (yiu_any_type(),)
        return inputs

    RETURN_TYPES = yiu_bypass_type_tuple(tuple([yiu_any_type()] * yiu_max_flow_num()))
    RETURN_NAMES = yiu_bypass_type_tuple(tuple([f"value{i}" for i in range(yiu_max_flow_num())]))
    FUNCTION = "while_loop_close"

    def _candidate_ids(self, node_id, dynprompt):
        candidates = []
        raw = str(node_id)
        candidates.append(raw)
        try:
            display_id = dynprompt.get_display_node_id(raw)
            if display_id is not None:
                candidates.append(str(display_id))
        except Exception:
            pass
        if "." in raw:
            parts = raw.split(".")
            for i in range(len(parts) - 1, 0, -1):
                candidates.append(".".join(parts[:i]))
            candidates.append(parts[0])

        unique = []
        seen = set()
        for c in candidates:
            if c not in seen:
                unique.append(c)
                seen.add(c)
        return unique

    def _resolve_node_id(self, node_id, dynprompt):
        for candidate in self._candidate_ids(node_id, dynprompt):
            try:
                dynprompt.get_node(candidate)
                return candidate
            except Exception:
                continue
        return None

    def _resolve_recurse_target_id(self, node_id, dynprompt):
        return self._resolve_node_id(node_id, dynprompt)

    def _resolve_output_count(self, node_id, dynprompt):
        resolved_node_id = self._resolve_node_id(node_id, dynprompt)
        if resolved_node_id is None:
            return yiu_max_flow_num()
        try:
            node_info = dynprompt.get_node(resolved_node_id)
            class_type = node_info.get("class_type")
            class_def = ALL_NODE_CLASS_MAPPINGS.get(class_type)
            return_types = getattr(class_def, "RETURN_TYPES", ()) if class_def is not None else ()
            if return_types:
                return len(return_types)
        except Exception:
            pass
        return yiu_max_flow_num()

    def explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
        resolved_node_id = self._resolve_node_id(node_id, dynprompt)
        if resolved_node_id is None:
            return
        node_info = dynprompt.get_node(resolved_node_id)
        if "inputs" not in node_info:
            return

        for _, v in node_info["inputs"].items():
            if is_link is not None and is_link(v):
                parent_id = self._resolve_node_id(v[0], dynprompt)
                if parent_id is None:
                    continue
                display_id = self._resolve_node_id(dynprompt.get_display_node_id(parent_id), dynprompt) or parent_id
                display_node = dynprompt.get_node(display_id)
                class_type = display_node["class_type"]
                if class_type not in ["yiuWhileLoopEnd"]:
                    parent_ids.append(display_id)
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream, parent_ids)
                upstream[parent_id].append(resolved_node_id)

    def explore_output_nodes(self, dynprompt, upstream, output_nodes, parent_ids):
        for parent_id in upstream:
            display_id = self._resolve_node_id(dynprompt.get_display_node_id(parent_id), dynprompt) or parent_id
            for output_id in output_nodes:
                node_id = output_nodes[output_id][0]
                if node_id in parent_ids and display_id == node_id and output_id not in upstream[parent_id]:
                    upstream[parent_id].append(output_id)

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)

    def _resolve_display_id(self, node_id, dynprompt):
        raw = str(node_id)
        if dynprompt is None:
            return raw
        try:
            display_id = dynprompt.get_display_node_id(raw)
            if display_id is not None:
                return str(display_id)
        except Exception:
            pass
        if "." in raw:
            return raw.split(".")[0]
        return raw

    def while_loop_close(self, flow, condition, recurse_target="", dynprompt=None, unique_id=None, **kwargs):
        if not condition:
            values = []
            for i in range(yiu_max_flow_num()):
                values.append(kwargs.get(f"initial_value{i}", None))
            return tuple(values)

        _debug_print(
            "[yiu_while_loop] while_close entry"
            + " unique_id="
            + str(unique_id)
            + " flow="
            + str(flow)
            + " recurse_target_in="
            + str(recurse_target)
            + " initial_value0="
            + str(kwargs.get("initial_value0", None))
        )

        # #region debug-point A:while-close-entry
        _report_debug_event(
            "A",
            "internal.py:while_loop_close:entry",
            "[DEBUG] while_loop_close entered",
            {
                "unique_id": str(unique_id),
                "flow": str(flow),
                "condition": bool(condition),
                "initial_value0": None if kwargs.get("initial_value0", None) is None else str(kwargs.get("initial_value0")),
                "initial_value1_is_none": kwargs.get("initial_value1", None) is None,
                "initial_value3": None if kwargs.get("initial_value3", None) is None else str(kwargs.get("initial_value3")),
                "initial_value4": None if kwargs.get("initial_value4", None) is None else str(kwargs.get("initial_value4")),
            },
        )
        # #endregion

        provided_recurse_target = recurse_target if isinstance(recurse_target, str) and recurse_target.strip() else None
        if provided_recurse_target is not None:
            recurse_target = self._resolve_node_id(provided_recurse_target, dynprompt) or str(provided_recurse_target)
        else:
            recurse_target = self._resolve_recurse_target_id(unique_id, dynprompt) or str(unique_id)
        upstream = {}
        parent_ids = []
        self.explore_dependencies(recurse_target, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))

        prompts = dynprompt.get_original_prompt()
        output_nodes = {}
        for node_id in prompts:
            node = prompts[node_id]
            if "inputs" not in node:
                continue
            class_type = node["class_type"]
            class_def = ALL_NODE_CLASS_MAPPINGS.get(class_type)
            if class_def is None:
                continue
            if hasattr(class_def, "OUTPUT_NODE") and class_def.OUTPUT_NODE is True:
                for _, v in node["inputs"].items():
                    if is_link is not None and is_link(v):
                        output_nodes[node_id] = v

        graph = GraphBuilder()
        self.explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids)
        contained = {}
        open_node_raw = str(flow[0])
        open_node_display = self._resolve_display_id(open_node_raw, dynprompt)
        open_node = self._resolve_node_id(open_node_display, dynprompt) or open_node_display
        self.collect_contained(open_node, upstream, contained)
        resolved_unique_id = self._resolve_node_id(recurse_target, dynprompt) or str(recurse_target)
        contained[resolved_unique_id] = True
        contained[open_node] = True

        _debug_print(
            "[yiu_while_loop] while_close resolved"
            + " open_node="
            + str(open_node)
            + " open_node_raw="
            + str(open_node_raw)
            + " recurse_target="
            + str(recurse_target)
            + " resolved_unique_id="
            + str(resolved_unique_id)
            + " contained_count="
            + str(len(contained))
        )

        # #region debug-point A:while-close-resolve
        _report_debug_event(
            "A",
            "internal.py:while_loop_close:resolve",
            "[DEBUG] while_loop_close resolved recursion nodes",
            {
                "unique_id": str(unique_id),
                "recurse_target": str(recurse_target),
                "resolved_unique_id": str(resolved_unique_id),
                "open_node": str(open_node),
                "parent_ids": [str(x) for x in parent_ids],
                "contained": [str(x) for x in contained.keys()],
            },
        )
        # #endregion

        for node_id in list(contained.keys()):
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == resolved_unique_id else node_id)
            node.set_override_display_id(node_id)

        for node_id in list(contained.keys()):
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == resolved_unique_id else node_id)
            for k, v in original_node["inputs"].items():
                resolved_parent_id = self._resolve_node_id(v[0], dynprompt) if (is_link is not None and is_link(v)) else None
                if resolved_parent_id is not None and resolved_parent_id in contained:
                    parent = graph.lookup_node("Recurse" if resolved_parent_id == resolved_unique_id else resolved_parent_id)
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)

        new_open = graph.lookup_node(open_node)
        for i in range(yiu_max_flow_num()):
            key = f"initial_value{i}"
            value = kwargs.get(key, None)
            if value is None:
                continue
            new_open.set_input(key, value)

        # Allow nodes like yiuUpscaleLoopStart to receive loop-carried state
        # through dedicated internal inputs instead of hidden inputs.
        if "initial_value0" in kwargs and kwargs.get("initial_value0", None) is not None:
            new_open.set_input("_loop_index_in", kwargs.get("initial_value0"))
        if "initial_value1" in kwargs and kwargs.get("initial_value1", None) is not None:
            new_open.set_input("_loop_image_in", kwargs.get("initial_value1"))

        # #region debug-point A:while-close-writeback
        _report_debug_event(
            "A",
            "internal.py:while_loop_close:writeback",
            "[DEBUG] while_loop_close wrote loop-carried inputs",
            {
                "open_node": str(open_node),
                "initial_value0": None if kwargs.get("initial_value0", None) is None else str(kwargs.get("initial_value0")),
                "initial_value1_is_none": kwargs.get("initial_value1", None) is None,
                "has_loop_index_writeback": kwargs.get("initial_value0", None) is not None,
                "has_loop_image_writeback": kwargs.get("initial_value1", None) is not None,
            },
        )
        # #endregion

        my_clone = graph.lookup_node("Recurse")
        recurse_output_count = max(0, int(self._resolve_output_count(resolved_unique_id, dynprompt)))
        # #region debug-point A:while-close-output-count
        _report_debug_event(
            "A",
            "internal.py:while_loop_close:output-count",
            "[DEBUG] while_loop_close resolved recurse output count",
            {
                "resolved_unique_id": str(resolved_unique_id),
                "recurse_output_count": int(recurse_output_count),
                "max_flow_num": int(yiu_max_flow_num()),
            },
        )
        # #endregion
        result = []
        for i in range(yiu_max_flow_num()):
            if i < recurse_output_count:
                result.append(my_clone.out(i))
            else:
                result.append(None)
        # #region debug-point A:while-close-return
        _report_debug_event(
            "A",
            "internal.py:while_loop_close:return",
            "[DEBUG] while_loop_close prepared recurse outputs",
            {
                "resolved_unique_id": str(resolved_unique_id),
                "returned_slots": int(len(result)),
                "non_none_slots": int(sum(1 for item in result if item is not None)),
            },
        )
        # #endregion
        return {"result": tuple(result), "expand": graph.finalize()}


class yiuMathInt:
    CATEGORY = "yiu_nodes/_internal"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -999999999, "max": 999999999}),
                "b": ("INT", {"default": 0, "min": -999999999, "max": 999999999}),
                "operation": (["add", "subtract"],),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "int_math_operation"

    def int_math_operation(self, a, b, operation):
        if operation == "subtract":
            return (int(a - b),)
        return (int(a + b),)


class yiuCompare:
    CATEGORY = "yiu_nodes/_internal"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "a": (yiu_any_type(), {"default": 0}),
                "b": (yiu_any_type(), {"default": 0}),
                "comparison": (["a < b", "a <= b", "a == b", "a != b", "a > b", "a >= b"], {"default": "a < b"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "compare"

    def compare(self, a=0, b=0, comparison="a < b"):
        if comparison == "a <= b":
            return (a <= b,)
        if comparison == "a == b":
            return (a == b,)
        if comparison == "a != b":
            return (a != b,)
        if comparison == "a > b":
            return (a > b,)
        if comparison == "a >= b":
            return (a >= b,)
        return (a < b,)


class yiuTilingMeta(YiuBaseNode):
    CATEGORY = "_internal"
    tooltips_dict = {
        "zh-CN": {
            "input": {
                "image": "输入图像（用于计算 rows/cols 等）",
                "max_tile_size": "tile 最大尺寸（px）。设为 0 表示关闭 tiling",
                "min_tile_overlap_px": "最小接缝宽度（px）",
                "tile_overlap_ratio": "接缝比例（0~1）",
            },
            "output": {
                "orig_h": "原图高度（px）",
                "orig_w": "原图宽度（px）",
                "rows": "行数",
                "cols": "列数",
                "overlap_x": "接缝宽度 X（px）",
                "overlap_y": "接缝宽度 Y（px）",
                "help_text": "帮助信息",
            },
        },
        "en": {
            "input": {
                "image": "Input image (for computing rows/cols, etc.)",
                "max_tile_size": "Max tile size (px). Set 0 to disable tiling",
                "min_tile_overlap_px": "Minimum overlap width (px)",
                "tile_overlap_ratio": "Overlap ratio (0~1)",
            },
            "output": {
                "orig_h": "Original height (px)",
                "orig_w": "Original width (px)",
                "rows": "Rows",
                "cols": "Cols",
                "overlap_x": "Overlap X (px)",
                "overlap_y": "Overlap Y (px)",
                "help_text": "Help text",
            },
        },
    }

    base_inputs = {
        "image": ("IMAGE", {}),
        "max_tile_size": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION, "unit": "px"}),
        "min_tile_overlap_px": ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "unit": "px"}),
        "tile_overlap_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
    }

    base_outputs = {
        "orig_h": "INT",
        "orig_w": "INT",
        "rows": "INT",
        "cols": "INT",
        "overlap_x": "INT",
        "overlap_y": "INT",
        "help_text": "STRING",
    }

    MAIN = "tiling_meta"

    @classmethod
    def DISPLAY_NAME(cls):
        return "🧩 Tiling Meta"

    @classmethod
    def get_help_text(self):
        lang = get_current_language()
        help_texts = {
            "zh-CN": "📖 根据输入图像与 tiling 参数计算 rows/cols/overlap（供 tiling/untile 使用）。",
            "en": "📖 Computes rows/cols/overlap from the input image and tiling parameters (for tiling/untile).",
        }
        return get_localized_tooltips(self.tooltips_dict, lang) + "\n" + help_texts.get(lang, help_texts["en"])

    @classmethod
    def tiling_meta(cls, image, max_tile_size, min_tile_overlap_px, tile_overlap_ratio):
        _, h, w, _ = image.shape
        rows, cols, overlap_x, overlap_y = compute_tiling_params(
            int(h), int(w), yiu_int(max_tile_size, 0), yiu_int(min_tile_overlap_px, 0), yiu_float(tile_overlap_ratio, 0.0)
        )
        return (int(h), int(w), int(rows), int(cols), int(overlap_x), int(overlap_y))


class yiuImageTile(YiuBaseNode):
    CATEGORY = "_internal"
    tooltips_dict = {
        "zh-CN": {
            "input": {
                "image": "输入图像（整图）",
                "orig_h": "原图高度（px）",
                "orig_w": "原图宽度（px）",
                "rows": "行数",
                "cols": "列数",
                "overlap_x": "接缝宽度 X（px）",
                "overlap_y": "接缝宽度 Y（px）",
            },
            "output": {
                "tiles": "tiles 批量（IMAGE batch）",
                "help_text": "帮助信息",
            },
        },
        "en": {
            "input": {
                "image": "Input image (full image)",
                "orig_h": "Original height (px)",
                "orig_w": "Original width (px)",
                "rows": "Rows",
                "cols": "Cols",
                "overlap_x": "Overlap X (px)",
                "overlap_y": "Overlap Y (px)",
            },
            "output": {
                "tiles": "Tiles batch (IMAGE batch)",
                "help_text": "Help text",
            },
        },
    }

    base_inputs = {
        "image": ("IMAGE", {}),
        "orig_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "unit": "px"}),
        "orig_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "unit": "px"}),
        "rows": ("INT", {"default": 1, "min": 1, "max": 9999}),
        "cols": ("INT", {"default": 1, "min": 1, "max": 9999}),
        "overlap_x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "unit": "px"}),
        "overlap_y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "unit": "px"}),
    }

    base_outputs = {
        "tiles": "IMAGE",
        "help_text": "STRING",
    }

    MAIN = "tile"

    @classmethod
    def DISPLAY_NAME(cls):
        return "🧩 Image Tile (yiu)"

    @classmethod
    def get_help_text(self):
        lang = get_current_language()
        help_texts = {
            "zh-CN": "📖 将整图切成 tiles 批量（用于循环中间的放大节点）。",
            "en": "📖 Splits the full image into a tiles batch (for the upscaler inside the loop).",
        }
        return get_localized_tooltips(self.tooltips_dict, lang) + "\n" + help_texts.get(lang, help_texts["en"])

    @classmethod
    def tile(cls, image, orig_h, orig_w, rows, cols, overlap_x, overlap_y):
        return (tile_image_batch(image, yiu_int(orig_h, 0), yiu_int(orig_w, 0), yiu_int(rows, 1), yiu_int(cols, 1), yiu_int(overlap_x, 0), yiu_int(overlap_y, 0)),)


class yiuImageUntile(YiuBaseNode):
    CATEGORY = "_internal"
    tooltips_dict = {
        "zh-CN": {
            "input": {
                "tiles": "tiles 批量（IMAGE batch）",
                "scale": "本步缩放倍率",
                "orig_h": "原图高度（px）",
                "orig_w": "原图宽度（px）",
                "rows": "行数",
                "cols": "列数",
                "overlap_x": "接缝宽度 X（px）",
                "overlap_y": "接缝宽度 Y（px）",
            },
            "output": {
                "image": "拼回的整图（IMAGE）",
                "help_text": "帮助信息",
            },
        },
        "en": {
            "input": {
                "tiles": "Tiles batch (IMAGE batch)",
                "scale": "Scale for this step",
                "orig_h": "Original height (px)",
                "orig_w": "Original width (px)",
                "rows": "Rows",
                "cols": "Cols",
                "overlap_x": "Overlap X (px)",
                "overlap_y": "Overlap Y (px)",
            },
            "output": {
                "image": "Merged full image (IMAGE)",
                "help_text": "Help text",
            },
        },
    }

    base_inputs = {
        "tiles": ("IMAGE", {}),
        "scale": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 64.0, "step": 0.001}),
        "orig_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "unit": "px"}),
        "orig_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "unit": "px"}),
        "rows": ("INT", {"default": 1, "min": 1, "max": 9999}),
        "cols": ("INT", {"default": 1, "min": 1, "max": 9999}),
        "overlap_x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "unit": "px"}),
        "overlap_y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "unit": "px"}),
    }

    base_outputs = {
        "image": "IMAGE",
        "help_text": "STRING",
    }

    MAIN = "untile"

    @classmethod
    def DISPLAY_NAME(cls):
        return "🧩 Image Untile (yiu)"

    @classmethod
    def get_help_text(self):
        lang = get_current_language()
        help_texts = {
            "zh-CN": "📖 将 tiles 批量按 rows/cols/overlap 拼回整图。",
            "en": "📖 Merges a tiles batch back to a full image using rows/cols/overlap.",
        }
        return get_localized_tooltips(self.tooltips_dict, lang) + "\n" + help_texts.get(lang, help_texts["en"])

    @classmethod
    def untile(cls, tiles, scale, orig_h, orig_w, rows, cols, overlap_x, overlap_y):
        return (
            untile_image_batch(
                tiles,
                yiu_float(scale, 1.0),
                yiu_int(orig_h, 0),
                yiu_int(orig_w, 0),
                yiu_int(rows, 1),
                yiu_int(cols, 1),
                yiu_int(overlap_x, 0),
                yiu_int(overlap_y, 0),
            ),
        )
