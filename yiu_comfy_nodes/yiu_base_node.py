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

    @classmethod
    def INPUT_TYPES(cls):
        lang = get_current_language()
        inputs_with_tooltips = build_types(cls.base_inputs, "input", cls.tooltips_dict, lang)
        print_debug("\n--- Debug: INPUT_TYPES result ---")
        print_debug(inputs_with_tooltips)
        return {"required": inputs_with_tooltips}

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
            return (cls.get_help_text(),)

        mainFn_callable = getattr(cls, cls.MAIN)
        results = mainFn_callable(*args, **kwargs)
        return results + (cls.get_help_text(),)
    

    CATEGORY = "yiu_nodes"    
    @classmethod
    def get_category(cls):
        category = getattr(cls, 'CATEGORY', YiuBaseNode.CATEGORY)
        
        if category == YiuBaseNode.CATEGORY:
            return YiuBaseNode.CATEGORY
        else:
            return f"{YiuBaseNode.CATEGORY}/{category}"