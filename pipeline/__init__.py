"""
Pipeline package for Portrait Video Enhancer.
Provides sequential, VRAM-optimized frame enhancement for human portrait videos.
Includes automatic compatibility monkeypatches for:
1. PyTorch 2.x and torchvision.transforms.functional_tensor
2. basicsr 1.4.2 Registry duplicate registration and suffix support
"""

import sys

# 1. Auto-patch torchvision.transforms.functional_tensor for PyTorch 2.x compatibility with basicsr
if "torchvision.transforms.functional_tensor" not in sys.modules:
    try:
        from torchvision.transforms import functional as _F
        sys.modules["torchvision.transforms.functional_tensor"] = _F
    except ImportError:
        pass

# 2. Auto-patch basicsr Registry to prevent duplicate assertions and support suffix kwarg
try:
    import basicsr.utils.registry as _b_reg

    def _safe_do_register(self, name, obj):
        self._obj_map[name] = obj

    def _safe_register(self, obj=None, suffix=None, **kwargs):
        if obj is None:
            def deco(func_or_class):
                name = func_or_class.__name__
                if suffix:
                    name = f"{name}_{suffix}"
                self._obj_map[name] = func_or_class
                return func_or_class
            return deco
        name = obj.__name__
        if suffix:
            name = f"{name}_{suffix}"
        self._obj_map[name] = obj
        return obj

    _b_reg.Registry._do_register = _safe_do_register
    _b_reg.Registry.register = _safe_register
except Exception:
    pass
