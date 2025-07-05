import os
import locale
import re

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