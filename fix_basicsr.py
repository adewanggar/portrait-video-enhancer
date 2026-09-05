"""
fix_basicsr.py
Applies compatibility patches for PyTorch 2.x and BasicSR / CodeFormer.
Run once during setup: python fix_basicsr.py
"""

import os
import sys


def apply_patches():
    # 1. Patch main basicsr package
    base_dir = None
    try:
        import basicsr
        base_dir = os.path.dirname(basicsr.__file__)
    except Exception:
        print("[INFO] basicsr is not installed yet. Installing with --no-build-isolation...")
        import subprocess
        res = subprocess.run([sys.executable, "-m", "pip", "install", "basicsr", "--no-build-isolation"])
        if res.returncode != 0:
            print("[INFO] Retrying basicsr installation from official github repository...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "git+https://github.com/XPixelGroup/BasicSR.git"])
        try:
            import basicsr
            base_dir = os.path.dirname(basicsr.__file__)
        except Exception:
            fallback = os.path.expanduser(r"~\AppData\Roaming\Python\Python310\site-packages\basicsr")
            if os.path.exists(fallback):
                base_dir = fallback

    if base_dir and os.path.exists(base_dir):
        # Patch __init__.py
        init_path = os.path.join(base_dir, "__init__.py")
        safe_init = '''# BasicSR clean inference init
from .archs import *
from .utils import *
from .version import __gitsha__, __version__

try:
    from .data import *
    from .losses import *
    from .metrics import *
    from .models import *
    from .ops import *
    from .train import *
except Exception:
    pass
'''
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(safe_init)
        print(f"[OK] Patched {init_path}")

        # Patch utils/registry.py
        reg_path = os.path.join(base_dir, "utils", "registry.py")
        clean_registry_code = '''# Clean patched registry for basicsr
class Registry():
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        self._obj_map[name] = obj

    def register(self, obj=None, suffix=None, **kwargs):
        if obj is None:
            def deco(func_or_class):
                name = func_or_class.__name__
                if suffix:
                    name = f"{name}_{suffix}"
                self._do_register(name, func_or_class)
                return func_or_class
            return deco
        name = obj.__name__
        if suffix:
            name = f"{name}_{suffix}"
        self._do_register(name, obj)
        return obj

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()

DATASET_REGISTRY = Registry('dataset')
ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')
DATASET_SAMPLER_REGISTRY = Registry('dataset_sampler')
PREFETCH_DATA_LOADER_REGISTRY = Registry('prefetch_data_loader')
'''
        with open(reg_path, "w", encoding="utf-8") as f:
            f.write(clean_registry_code)
        print(f"[OK] Patched {reg_path}")

        # Patch utils/__init__.py
        utils_init_path = os.path.join(base_dir, "utils", "__init__.py")
        if os.path.exists(utils_init_path):
            with open(utils_init_path, "r", encoding="utf-8") as f:
                u_txt = f.read()
            if "DiffJPEG" not in u_txt:
                u_txt += """
try:
    from .diffjpeg import DiffJPEG
    from .img_process_util import USMSharp
    from .color_util import *
    from .matlab_functions import *
    from .registry import *
except Exception:
    pass
"""
                with open(utils_init_path, "w", encoding="utf-8") as f:
                    f.write(u_txt)
                print(f"[OK] Patched {utils_init_path}")

        # Patch degradations.py
        deg_path = os.path.join(base_dir, "data", "degradations.py")
        if os.path.exists(deg_path):
            with open(deg_path, "r", encoding="utf-8") as f:
                deg_txt = f.read()
            deg_txt = deg_txt.replace(
                "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
                "from torchvision.transforms.functional import rgb_to_grayscale"
            )
            with open(deg_path, "w", encoding="utf-8") as f:
                f.write(deg_txt)
            print(f"[OK] Patched {deg_path}")
    else:
        print("[WARNING] Could not locate basicsr installation directory to patch.")

    # 2. Patch codeformer internal basicsr if present
    codeformer_dir = os.path.expanduser(r"~\AppData\Roaming\Python\Python310\site-packages\codeformer")
    cf_bsr_init = os.path.join(codeformer_dir, "basicsr", "__init__.py")
    if os.path.exists(cf_bsr_init):
        with open(cf_bsr_init, "w", encoding="utf-8") as f:
            f.write("# Patched codeformer basicsr init\nfrom .archs import *\nfrom .utils import *\n")
        print(f"[OK] Patched {cf_bsr_init}")

    print("\nAll patches applied successfully!")


if __name__ == "__main__":
    apply_patches()
