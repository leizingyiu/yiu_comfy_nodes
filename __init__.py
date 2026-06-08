import importlib
import os
import sys


_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _load_impl_package():
    errors = []
    for name in ("yiu_comfy_nodes_impl", "yiu_comfy_nodes"):
        try:
            return importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {exc!r}")
            continue
    raise ImportError(
        "Neither 'yiu_comfy_nodes_impl' nor 'yiu_comfy_nodes' could be imported. "
        + "; ".join(errors)
    )


_impl = _load_impl_package()

NODE_CLASS_MAPPINGS = getattr(_impl, "NODE_CLASS_MAPPINGS", {})
NODE_DISPLAY_NAME_MAPPINGS = getattr(_impl, "NODE_DISPLAY_NAME_MAPPINGS", {})

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
