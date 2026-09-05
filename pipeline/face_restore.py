"""
pipeline/face_restore.py
Face restoration module supporting CodeFormer and GFPGAN.
Runs in fp16 half-precision, processes 1 frame at a time, and cleans CUDA cache.
"""

import os
import gc
import cv2
import torch
import numpy as np
from typing import Optional

# Auto-apply compatibility patches
import pipeline  # noqa: F401
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper


class FaceRestorer:
    def __init__(
        self,
        method: str = "CodeFormer",
        fidelity_weight: float = 0.6,
        scale: float = 4.0,
        weights_dir: str = "weights",
        half: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.method = method
        self.fidelity_weight = float(np.clip(fidelity_weight, 0.0, 1.0))
        self.scale = float(scale)
        self.weights_dir = weights_dir
        self.half = False  # Face models run in FP32 on GPU (~1.3GB VRAM) to preserve codebook precision
        self.device = torch.device(device)

        self.net = None
        self.face_helper = None
        self.gfpgan_restorer = None

        self._init_models()

    def _init_models(self):
        facelib_path = os.path.join(self.weights_dir, "facelib")

        if self.method == "GFPGAN":
            from gfpgan import GFPGANer
            gfpgan_weight = os.path.join(self.weights_dir, "gfpgan", "GFPGANv1.4.pth")
            if not os.path.exists(gfpgan_weight):
                raise FileNotFoundError(f"GFPGAN weight not found: {gfpgan_weight}")

            self.gfpgan_restorer = GFPGANer(
                model_path=gfpgan_weight,
                upscale=max(1, int(round(self.scale))),
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device,
            )
        else:
            # Default: CodeFormer
            from codeformer.basicsr.archs.codeformer_arch import CodeFormer as CodeFormerNet
            codeformer_weight = os.path.join(self.weights_dir, "CodeFormer", "codeformer.pth")
            if not os.path.exists(codeformer_weight):
                raise FileNotFoundError(f"CodeFormer weight not found: {codeformer_weight}")

            self.net = CodeFormerNet(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(self.device)

            checkpoint = torch.load(codeformer_weight, map_location=self.device)
            state_dict = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
            self.net.load_state_dict(state_dict)
            self.net.eval()

            # RetinaFace + ParseNet FaceRestoreHelper
            self.face_helper = FaceRestoreHelper(
                upscale_factor=self.scale,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                use_parse=True,
                device=self.device,
                model_rootpath=facelib_path,
            )

    def restore_face(
        self,
        img_bgr: np.ndarray,
        upsampled_canvas: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Detect, restore face(s), and composite onto upsampled_canvas or original frame.
        Clears CUDA working tensors per frame.
        """
        with torch.inference_mode():
            if self.method == "GFPGAN":
                return self._restore_gfpgan(img_bgr, upsampled_canvas)
            return self._restore_codeformer(img_bgr, upsampled_canvas)

    def _restore_codeformer(
        self,
        img_bgr: np.ndarray,
        upsampled_canvas: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self.face_helper.clean_all()
        self.face_helper.read_image(img_bgr)
        # Detect landmarks (avoid upscaling low-res input frames)
        det_resize = min(img_bgr.shape[1], 480)
        self.face_helper.get_face_landmarks_5(only_center_face=False, resize=det_resize, eye_dist_threshold=5)
        # Align & warp face crops
        self.face_helper.align_warp_face()

        if len(self.face_helper.cropped_faces) == 0:
            # No face found, return canvas or original
            return upsampled_canvas if upsampled_canvas is not None else img_bgr

        # Restore each detected face
        for cropped_face in self.face_helper.cropped_faces:
            # Normalize to [-1, 1] RGB
            face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            face_t = face_t.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.net(face_t, w=self.fidelity_weight, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

            del face_t, output
            restored_face = restored_face.astype("uint8")
            self.face_helper.add_restored_face(restored_face)

        # Composite face back
        self.face_helper.get_inverse_affine(None)
        if upsampled_canvas is not None:
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=upsampled_canvas)
        else:
            restored_img = self.face_helper.paste_faces_to_input_image()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return restored_img

    def _restore_gfpgan(
        self,
        img_bgr: np.ndarray,
        upsampled_canvas: Optional[np.ndarray] = None
    ) -> np.ndarray:
        _, _, restored_img = self.gfpgan_restorer.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )

        if upsampled_canvas is not None and restored_img.shape != upsampled_canvas.shape:
            restored_img = cv2.resize(
                restored_img,
                (upsampled_canvas.shape[1], upsampled_canvas.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return restored_img

    def offload_to_cpu(self):
        """Offload face model to CPU to conserve VRAM."""
        if self.net is not None:
            self.net.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def reload_to_gpu(self):
        """Reload face model to GPU."""
        if self.net is not None:
            self.net.to(self.device)
