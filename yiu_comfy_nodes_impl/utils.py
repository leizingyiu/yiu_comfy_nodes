import os
import locale
import re
import math

try:
    import torch
except Exception:
    torch = None

def print_once(msg):
    if not hasattr(print_once, 'printed'):
        print(msg)
        print_once.printed = True

def get_current_language():
    lang_dir = os.path.join(os.path.dirname(__file__), "lang_setting")
    
    lang_files = os.listdir(lang_dir) if os.path.exists(lang_dir) else []
    cn_file = None
    en_file = None

    for file_name in lang_files:
        if re.match(r"^cn\s*-\s*", file_name, re.IGNORECASE):
            cn_file = file_name
        elif re.match(r"^en\s*-\s*", file_name, re.IGNORECASE):
            en_file = file_name

    if cn_file and not en_file:
        return "zh-CN"
    elif en_file and not cn_file:
        return "en"

    print_once("⚙️ 当前语言未进行设置，可在 yiu_comfy_nodes/lang_setting 文件夹，删除指定语言的 txt 文件来切换语言。")

    try:
        system_locale = locale.getdefaultlocale()
        lang_code = system_locale[0] if system_locale and system_locale[0] else None
        if lang_code:
            if lang_code.startswith("zh"):
                return "zh-CN"
            elif lang_code.startswith("en"):
                return "en"
            else:
                return "en" 
    except:
        pass

    return "en"




def build_types(base_dict, dict_type, tooltips_dict, lang):
    lang_tooltips = tooltips_dict.get(lang, tooltips_dict.get("en", {})).get(dict_type, {})

    if dict_type == "input":
        result = {}
        for key, value in base_dict.items():
            typ, cfg = value
            cfg_copy = cfg.copy() if isinstance(cfg, dict) else {}
            tooltip = lang_tooltips.get(key, "")

            unit = cfg_copy.get("unit", None)
            if unit:
                cfg_copy["ui_name"] = f"{key} ({unit})"
                cfg_copy.pop("unit")

            if tooltip:
                cfg_copy["description"] = tooltip

            result[key] = (typ, cfg_copy)
        return result

    elif dict_type == "output":
        final_output_definition = {}
        for key, typ in base_dict.items():
            description = lang_tooltips.get(key, "")
            final_output_definition[key] = (typ, {"description": description})
        return final_output_definition


def get_localized_tooltips(tooltips_dict: dict, lang: str) -> str:
    
    if lang not in tooltips_dict:
        return f"错误：语言 '{lang}' 不存在于 tooltips_dict 中。"

    localized_data = tooltips_dict[lang]
    formatted_output = []

    for category, items in localized_data.items():
        formatted_output.append(f"--- {category.upper()} ---")
        for key, value in items.items():
            formatted_output.append(f"{key}: {value}")
        formatted_output.append("") 

    return "\n".join(formatted_output)


class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


class TautologyStr(str):
    def __ne__(self, other):
        return False


class ByPassTypeTuple(tuple):
    def __getitem__(self, index):
        if index > 0:
            index = 0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return TautologyStr(item)
        return item


YIU_MAX_FLOW_NUM = 20
YIU_ANY_TYPE = AlwaysEqualProxy("*")


def yiu_max_flow_num():
    return YIU_MAX_FLOW_NUM


def yiu_any_type():
    return YIU_ANY_TYPE


def yiu_bypass_type_tuple(value):
    return ByPassTypeTuple(value)


def yiu_scalar(x):
    if x is None:
        return None
    if isinstance(x, (int, float, bool)):
        return x
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return yiu_scalar(x[0])
    if torch is not None and isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got numel={int(x.numel())}, shape={tuple(x.shape)}")
        return x.item()
    return x


def yiu_int(x, default=0):
    v = yiu_scalar(x)
    if v is None:
        return int(default)
    return int(v)


def yiu_float(x, default=0.0):
    v = yiu_scalar(x)
    if v is None:
        return float(default)
    return float(v)


def derive_target_size(w, h, target_width, target_height):
    if target_width <= 0 and target_height <= 0:
        return w, h
    if target_width <= 0:
        tw = int(round(target_height * (w / max(1, h))))
        return max(1, tw), max(1, int(target_height))
    if target_height <= 0:
        th = int(round(target_width * (h / max(1, w))))
        return max(1, int(target_width)), max(1, th)
    return max(1, int(target_width)), max(1, int(target_height))


def compute_step_total(w, h, target_w, target_h, scale_every_time):
    total_scale = max(target_w / max(1, w), target_h / max(1, h))
    if total_scale <= 1.0:
        return float(total_scale), 1
    s = float(scale_every_time)
    if not math.isfinite(s) or s <= 1.0:
        return float(total_scale), 1
    steps = int(math.ceil(math.log(total_scale) / math.log(s)))
    return float(total_scale), max(1, steps)


def compute_scale_this_time(total_scale, scale_every_time, step_total, step_index):
    if step_total <= 1:
        return 1.0 if total_scale <= 1.0 else float(total_scale)
    s = float(scale_every_time)
    i = int(step_index)
    if i >= step_total - 1:
        scale = float(total_scale) / (s ** max(0, i))
    else:
        scale = s
    if not math.isfinite(scale) or scale <= 1.0:
        return 1.0
    return float(scale)


def ceil_div(a, b):
    return (a + b - 1) // b


def compute_tiling_params(h, w, max_tile_size, min_tile_overlap_px, tile_overlap_ratio):
    ts = int(max_tile_size)
    if ts <= 0:
        return 1, 1, 0, 0

    rows = max(1, int(ceil_div(int(h), ts)))
    cols = max(1, int(ceil_div(int(w), ts)))

    overlap = int(max(int(min_tile_overlap_px), round(ts * float(tile_overlap_ratio))))
    if overlap < 0:
        overlap = 0
    if overlap >= ts:
        overlap = max(0, ts - 1)

    return rows, cols, overlap, overlap


def tile_image_batch(image, orig_h, orig_w, rows, cols, overlap_x, overlap_y):
    b, h, w, c = image.shape
    rh = int(orig_h) if int(orig_h) > 0 else int(h)
    rw = int(orig_w) if int(orig_w) > 0 else int(w)
    r = max(1, int(rows))
    col = max(1, int(cols))
    ox = max(0, int(overlap_x))
    oy = max(0, int(overlap_y))

    if r == 1 and col == 1:
        return image

    core_h = int((rh + r - 1) // r)
    core_w = int((rw + col - 1) // col)

    tiles = []
    sizes = []
    for bi in range(int(b)):
        img = image[bi]
        for yi in range(r):
            y0 = yi * core_h
            y1 = min(rh, (yi + 1) * core_h)
            y0e = y0 - oy if yi > 0 else y0
            y1e = y1 + oy if yi < r - 1 else y1
            y0e = max(0, int(y0e))
            y1e = min(rh, int(y1e))

            for xi in range(col):
                x0 = xi * core_w
                x1 = min(rw, (xi + 1) * core_w)
                x0e = x0 - ox if xi > 0 else x0
                x1e = x1 + ox if xi < col - 1 else x1
                x0e = max(0, int(x0e))
                x1e = min(rw, int(x1e))

                tile = img[y0e:y1e, x0e:x1e, :]
                tiles.append(tile)
                sizes.append((int(y1e - y0e), int(x1e - x0e)))

    max_h = max((s[0] for s in sizes), default=int(h))
    max_w = max((s[1] for s in sizes), default=int(w))

    out = image.new_zeros((len(tiles), max_h, max_w, c))
    for i, t in enumerate(tiles):
        th, tw, _ = t.shape
        out[i, :th, :tw, :] = t

    return out


def _torch_linspace(a, b, steps, device, dtype):
    if torch is None:
        raise RuntimeError("torch is required for untile_image_batch")
    return torch.linspace(float(a), float(b), int(steps), device=device, dtype=dtype)


def untile_image_batch(tiles, scale, orig_h, orig_w, rows, cols, overlap_x, overlap_y):
    if torch is None:
        raise RuntimeError("torch is required for untile_image_batch")

    r = max(1, int(rows))
    col = max(1, int(cols))
    if r == 1 and col == 1:
        return tiles[:1]

    rh = int(orig_h)
    rw = int(orig_w)
    if rh <= 0 or rw <= 0:
        raise ValueError("orig_h/orig_w must be provided for untile.")

    s = float(scale)
    if not math.isfinite(s) or s <= 0.0:
        s = 1.0

    per_image = r * col
    if int(tiles.shape[0]) % per_image != 0:
        raise ValueError(f"tiles batch size ({int(tiles.shape[0])}) is not divisible by rows*cols ({per_image}).")
    batch = int(tiles.shape[0]) // per_image
    c = int(tiles.shape[3])
    core_h = int((rh + r - 1) // r)
    core_w = int((rw + col - 1) // col)
    device = tiles.device
    dtype = tiles.dtype

    out_h = max(1, int(round(rh * s)))
    out_w = max(1, int(round(rw * s)))

    merged = []
    idx = 0
    for _ in range(batch):
        canvas = torch.zeros((out_h, out_w, c), device=device, dtype=dtype)
        weight = torch.zeros((out_h, out_w, 1), device=device, dtype=dtype)

        for yi in range(r):
            y0 = yi * core_h
            y1 = min(rh, (yi + 1) * core_h)
            oy0 = int(overlap_y)
            y0e = y0 - oy0 if yi > 0 else y0
            y1e = y1 + oy0 if yi < r - 1 else y1
            y0e = max(0, int(y0e))
            y1e = min(rh, int(y1e))

            for xi in range(col):
                x0 = xi * core_w
                x1 = min(rw, (xi + 1) * core_w)
                ox0 = int(overlap_x)
                x0e = x0 - ox0 if xi > 0 else x0
                x1e = x1 + ox0 if xi < col - 1 else x1
                x0e = max(0, int(x0e))
                x1e = min(rw, int(x1e))

                y0es = int(round(y0e * s))
                y1es = int(round(y1e * s))
                x0es = int(round(x0e * s))
                x1es = int(round(x1e * s))

                y0es = max(0, min(out_h, int(y0es)))
                y1es = max(0, min(out_h, int(y1es)))
                x0es = max(0, min(out_w, int(x0es)))
                x1es = max(0, min(out_w, int(x1es)))

                tile_h = int(tiles.shape[1])
                tile_w = int(tiles.shape[2])
                y1es = max(y0es, min(y1es, y0es + tile_h))
                x1es = max(x0es, min(x1es, x0es + tile_w))

                ths = max(0, int(y1es - y0es))
                tws = max(0, int(x1es - x0es))
                if ths <= 0 or tws <= 0:
                    idx += 1
                    continue

                t = tiles[idx, :ths, :tws, :]
                idx += 1

                wy = torch.ones((ths,), device=device, dtype=dtype)
                wx = torch.ones((tws,), device=device, dtype=dtype)

                oy = max(0, int(round(int(oy0) * s)))
                ox = max(0, int(round(int(ox0) * s)))

                if oy > 0 and yi > 0:
                    wy[: min(oy, ths)] = _torch_linspace(0.0, 1.0, min(oy, ths), device, dtype)
                if oy > 0 and yi < r - 1:
                    wy[-min(oy, ths) :] = _torch_linspace(1.0, 0.0, min(oy, ths), device, dtype)

                if ox > 0 and xi > 0:
                    wx[: min(ox, tws)] = _torch_linspace(0.0, 1.0, min(ox, tws), device, dtype)
                if ox > 0 and xi < col - 1:
                    wx[-min(ox, tws) :] = _torch_linspace(1.0, 0.0, min(ox, tws), device, dtype)

                w2d = (wy[:, None] * wx[None, :])[:, :, None]
                canvas[y0es:y1es, x0es:x1es, :] += t * w2d
                weight[y0es:y1es, x0es:x1es, :] += w2d

        out = canvas / torch.clamp(weight, min=1e-6)
        merged.append(out[None, ...])

    return torch.cat(merged, dim=0)
