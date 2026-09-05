"""
pipeline/upscale.py
Tiled Real-ESRGAN wrapper running in fp16 with sequential memory offloading.
Optimized for 6GB VRAM on NVIDIA RTX 3050.
"""

import os
import gc
import cv2
import torch
import numpy as np
from typing import Optional

# Auto-apply torchvision fix before importing basicsr
import pipeline  # noqa: F401


class RealESRGANUpscaler:
    def __init__(
        self,
        model_name: str = "RealESRGAN_x4plus",
        weights_dir: str = "weights/realesrgan",
        tile: int = 400,
        tile_pad: int = 10,
        half: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_name = model_name
        self.weights_dir = weights_dir
        self.tile = tile
        self.tile_pad = tile_pad
        self.half = half and (device == "cuda")
        self.device = torch.device(device)
        self.scale = 2 if "x2plus" in model_name else 4
        self.upsampler = None
        self._init_model()

    def _init_model(self):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_path = os.path.join(self.weights_dir, f"{self.model_name}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weight not found at {model_path}. Run python download_models.py first."
            )

        if self.scale == 2:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
        else:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )

        self.upsampler = RealESRGANer(
            scale=self.scale,
            model_path=model_path,
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=0,
            half=self.half,
            device=self.device,
        )

    def enhance_frame(self, img_bgr: np.ndarray, outscale: Optional[float] = None) -> np.ndarray:
        """
        Enhance a single BGR image frame.
        Performs uint8 quantization directly on GPU tensor, completely bypassing
        the 3x float32 CPU RAM allocations that cause out-of-memory errors on large frames.
        """
        if self.upsampler is None:
            self._init_model()

        h_input, w_input = img_bgr.shape[0:2]
        upsampler = self.upsampler

        # Convert BGR uint8 to RGB float32 [0, 1]
        img = img_bgr.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            upsampler.pre_process(img)
            if upsampler.tile_size > 0:
                upsampler.tile_process()
            else:
                upsampler.process()

            output_tensor = upsampler.post_process()

            # In-place quantization directly on GPU: 0 extra bytes allocated
            output_tensor.data.clamp_(0, 1).mul_(255.0).round_()
            out_uint8 = output_tensor.data.squeeze(0).to(torch.uint8)
            # Convert RGB to BGR and permute (C, H, W) -> (H, W, C)
            out_bgr = out_uint8[[2, 1, 0], :, :].permute(1, 2, 0).cpu().numpy()

            del output_tensor, out_uint8, img
            if hasattr(upsampler, "output"):
                del upsampler.output
            if hasattr(upsampler, "img"):
                del upsampler.img
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        target_scale = float(outscale) if outscale is not None else float(self.scale)
        if target_scale != float(self.scale):
            target_w = int(round(w_input * target_scale))
            target_h = int(round(h_input * target_scale))
            out_bgr = cv2.resize(
                out_bgr,
                (target_w, target_h),
                interpolation=cv2.INTER_LANCZOS4 if target_scale > float(self.scale) else cv2.INTER_AREA,
            )

        return out_bgr

    def offload_to_cpu(self):
        """Offload model to CPU to free VRAM for face restoration."""
        if self.upsampler is not None and hasattr(self.upsampler, "model"):
            self.upsampler.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    def reload_to_gpu(self):
        """Reload model back to GPU."""
        if self.upsampler is not None and hasattr(self.upsampler, "model"):
            self.upsampler.model.to(self.device)
            if self.half:
                self.upsampler.model.half()
