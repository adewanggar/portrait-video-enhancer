"""
pipeline/interpolate.py
Optional frame interpolation module using RIFE (Real-Time Intermediate Flow Estimation)
for 2x frame rate motion smoothing.
"""

import os
import torch
import numpy as np
import cv2
from typing import List, Generator, Tuple


class RIFEInterpolater:
    def __init__(
        self,
        weights_path: str = "weights/rife/flownet.pkl",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        half: bool = True,
    ):
        self.weights_path = weights_path
        self.device = torch.device(device)
        self.half = half and (device == "cuda")
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"RIFE weight not found at {self.weights_path}. "
                "Download it with: python download_models.py --category rife"
            )

        checkpoint = torch.load(self.weights_path, map_location=self.device)
        if hasattr(checkpoint, "eval"):
            # Checkpoint is an already pickled nn.Module
            self.model = checkpoint
        elif isinstance(checkpoint, dict):
            # State dict - try to load standard IFNet architecture
            from pipeline.rife_arch import IFNet
            self.model = IFNet()
            state = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            self.model.load_state_dict(state, strict=False)
        else:
            self.model = checkpoint

        self.model.eval()
        self.model.to(self.device)
        if self.half:
            self.model.half()

    def interpolate_pair(self, img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> np.ndarray:
        """
        Generate intermediate frame between img1 and img2.
        Input: BGR uint8 images.
        Output: Intermediate BGR uint8 image.
        """
        h, w, _ = img1_bgr.shape
        # Pad to multiple of 32
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        pad = (0, pw - w, 0, ph - h)

        t1 = torch.from_numpy(img1_bgr.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        t2 = torch.from_numpy(img2_bgr.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

        if pad[1] > 0 or pad[3] > 0:
            t1 = torch.nn.functional.pad(t1, pad)
            t2 = torch.nn.functional.pad(t2, pad)

        t1 = t1.to(self.device)
        t2 = t2.to(self.device)
        if self.half:
            t1 = t1.half()
            t2 = t2.half()

        with torch.no_grad():
            if hasattr(self.model, "inference"):
                mid = self.model.inference(t1, t2)
            else:
                mid = self.model(torch.cat((t1, t2), 1))
            if isinstance(mid, (tuple, list)):
                mid = mid[0]

        mid = mid[0, :, :h, :w].cpu().float().numpy().transpose(1, 2, 0)
        mid = np.clip(mid * 255.0, 0, 255).astype("uint8")

        del t1, t2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return mid

    def interpolate_sequence(
        self,
        frame_paths: List[str],
        output_dir: str
    ) -> List[str]:
        """
        Interpolate consecutive frame pairs, doubling the total frame count.
        Writes frames directly to disk to prevent RAM accumulation.
        """
        os.makedirs(output_dir, exist_ok=True)
        out_paths = []
        count = 0

        for i in range(len(frame_paths) - 1):
            f1 = cv2.imread(frame_paths[i])
            f2 = cv2.imread(frame_paths[i + 1])

            # Write current frame
            p1 = os.path.join(output_dir, f"frame_{count:06d}.png")
            cv2.imwrite(p1, f1)
            out_paths.append(p1)
            count += 1

            # Generate and write middle frame
            mid = self.interpolate_pair(f1, f2)
            p_mid = os.path.join(output_dir, f"frame_{count:06d}.png")
            cv2.imwrite(p_mid, mid)
            out_paths.append(p_mid)
            count += 1

        # Write last frame
        last_f = cv2.imread(frame_paths[-1])
        p_last = os.path.join(output_dir, f"frame_{count:06d}.png")
        cv2.imwrite(p_last, last_f)
        out_paths.append(p_last)

        return out_paths
