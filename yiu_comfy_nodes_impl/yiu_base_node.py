from .utils import get_current_language, build_types,get_localized_tooltips

DEBUG_MODE = False
def print_debug(msg):
    if DEBUG_MODE:
        from datetime import datetime
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}] {msg}")

        
class YiuBaseNode:
   
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    tooltips_dict = {}
    EXTRA_HELP_TEXT = ""

    tooltips_dict = {
        "zh-CN":{
            "input": {},
            "output":{}
            },
        "en":{
            "input": {},
            "output":{}
            }
        }

    @classmethod
    def get_extra_help_text(cls):
        extra = getattr(cls, "EXTRA_HELP_TEXT", "")
        if not extra:
            return ""
        return str(extra).strip()

    @classmethod
    def build_help_text(cls):
        help_text = cls.get_help_text()
        extra = cls.get_extra_help_text()
        if extra:
            return f"{help_text}\n\n---\n\n{extra}"
        return help_text


    @classmethod
    def INPUT_TYPES(cls):
        lang = get_current_language()
        required_inputs = build_types(getattr(cls, "base_inputs", {}), "input", cls.tooltips_dict, lang)
        optional_inputs = build_types(getattr(cls, "base_optional_inputs", {}), "input", cls.tooltips_dict, lang)
        hidden_inputs = getattr(cls, "base_hidden_inputs", {})

        print_debug("\n--- Debug: INPUT_TYPES result ---")
        print_debug({"required": required_inputs, "optional": optional_inputs, "hidden": hidden_inputs})

        result = {"required": required_inputs}
        if optional_inputs:
            result["optional"] = optional_inputs
        if hidden_inputs:
            result["hidden"] = hidden_inputs
        return result

    @classmethod
    def VALID_OUTPUT_TYPES(cls):
        lang = get_current_language()
        final_output_definitions = build_types(cls.base_outputs, "output", cls.tooltips_dict, lang)
        print_debug("\n--- Debug: VALID_OUTPUT_TYPES result ---")
        print_debug(final_output_definitions) 
        return final_output_definitions


    def __init_subclass__(cls):
        output_defs = cls.VALID_OUTPUT_TYPES() 
        
        return_types_list = []
        return_names_list = []
        for name, (typ, _) in output_defs.items(): 
            return_names_list.append(name)
            return_types_list.append(typ)
        
        cls.RETURN_TYPES = tuple(return_types_list)
        cls.RETURN_NAMES = tuple(return_names_list)

        print_debug(f"\n--- Debug: {cls.__name__} initialized with ---")
        print_debug(f"RETURN_TYPES: {cls.RETURN_TYPES}")
        print_debug(f"RETURN_NAMES: {cls.RETURN_NAMES}")
        print_debug("---------------------------------------------")
        

    FUNCTION = "process"
    @classmethod
    def process(cls, *args, **kwargs):
        if not args and not kwargs:
            return (cls.build_help_text(),)

        mainFn_callable = getattr(cls, cls.MAIN)
        results = mainFn_callable(*args, **kwargs)
        help_text = cls.build_help_text()
        if isinstance(results, dict) and "result" in results:
            results_tuple = tuple(results.get("result", ()))
            results["result"] = results_tuple + (help_text,)
            return results
        return results + (help_text,)
    

    CATEGORY = "yiu_nodes"    
    @classmethod
    def get_category(cls):
        category = getattr(cls, 'CATEGORY', YiuBaseNode.CATEGORY)
        
        if category == YiuBaseNode.CATEGORY:
            return YiuBaseNode.CATEGORY
        else:
            return f"{YiuBaseNode.CATEGORY}/{category}"
