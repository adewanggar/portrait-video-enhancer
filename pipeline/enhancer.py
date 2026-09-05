"""
pipeline/enhancer.py
Master video processing pipeline for Portrait Video Enhancer.
Executes frame-by-frame enhancement with sequential memory offloading.
"""

import os
import gc
import cv2
import shutil
import tempfile
import numpy as np
import torch
from typing import Callable, Optional, Tuple, Dict, Any

from pipeline.extract import extract_frames_and_audio
from pipeline.upscale import RealESRGANUpscaler
from pipeline.face_restore import FaceRestorer
from pipeline.reassemble import reassemble_video


class VideoEnhancer:
    def __init__(
        self,
        face_restorer_type: str = "CodeFormer",
        fidelity_weight: float = 0.6,
        upscale_model: str = "RealESRGAN_x4plus",
        tile_size: int = 400,
        enable_rife: bool = False,
        mode: str = "Fast mode",
        sequential_offload: bool = False,
        weights_dir: str = "weights",
        target_resolution: str = "1080p (Full HD)",
        focus_mode: str = "Pakaian + Wajah (Lengkap)",
        fabric_sharpness: float = 0.35,
        color_boost: float = 0.15,
    ):
        self.face_restorer_type = face_restorer_type
        self.fidelity_weight = fidelity_weight
        self.upscale_model = upscale_model
        self.tile_size = tile_size if mode == "Fast mode" else min(tile_size, 300)
        self.enable_rife = enable_rife and (mode == "Quality mode" or enable_rife)
        self.mode = mode
        self.target_resolution = target_resolution
        self.focus_mode = focus_mode
        self.fabric_sharpness = float(fabric_sharpness)
        self.color_boost = float(color_boost)
        # 6GB VRAM is ample for both models simultaneously (~1.85GB total)
        # Only offload if explicitly requested or low VRAM (<4GB)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.sequential_offload = sequential_offload and (total_vram_gb < 4.5)
        else:
            self.sequential_offload = False
        self.weights_dir = weights_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scale = 2 if "x2plus" in upscale_model else 4

    def process_video(
        self,
        input_video_path: str,
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[str, str, str]:
        """
        Enhance human portrait video.
        Returns:
            output_video_path: path to final enhanced MP4 video.
            preview_before: path to middle frame before enhancement.
            preview_after: path to middle frame after enhancement.
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        # Use workspace directory on drive H: where 60+ GB free space is available
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_temp = os.path.join(workspace_root, ".temp_work")
        os.makedirs(base_temp, exist_ok=True)
        work_dir = tempfile.mkdtemp(prefix="portrait_enhancer_", dir=base_temp)
        enhanced_frames_dir = os.path.join(work_dir, "enhanced_frames")
        os.makedirs(enhanced_frames_dir, exist_ok=True)

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_video_path), "enhanced_outputs")
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(input_video_path))[0]
        output_video_path = os.path.join(output_dir, f"{base_name}_enhanced.mp4")

        try:
            # 1. Extract frames + audio + probe
            if progress_callback:
                progress_callback(0.05, "Extracting video frames and audio track...")

            meta = extract_frames_and_audio(input_video_path, work_dir)
            frame_files = meta.get("frame_files", [])
            total_frames = len(frame_files)

            if total_frames == 0:
                raise RuntimeError("No frames extracted from input video.")

            fps = meta.get("fps", 30.0)
            audio_path = meta.get("audio_path")

            # 2. Adaptive Tiling & Target Dimensions
            in_w = meta.get("width", 1920)
            in_h = meta.get("height", 1080)
            effective_tile = self.tile_size
            if max(in_w, in_h) <= 1280 or (in_w * in_h <= 1280 * 720):
                effective_tile = 0

            # Calculate safe target dimensions (prevents 5K/8K memory exhaustion on vertical video)
            max_dim = max(in_w, in_h)
            if "1080p" in self.target_resolution:
                target_max = 1920
            elif "720p" in self.target_resolution:
                target_max = 1280
            elif "2K" in self.target_resolution:
                target_max = 2560
            elif "4K" in self.target_resolution:
                target_max = 3840
            else:
                target_max = max_dim * self.scale

            target_scale = max(1.0, min(float(self.scale), target_max / float(max_dim)))
            target_w = (int(round(in_w * target_scale)) // 2) * 2
            target_h = (int(round(in_h * target_scale)) // 2) * 2
            eff_scale = float(target_scale)

            if progress_callback:
                progress_callback(0.10, f"Initializing models (Output: {target_w}x{target_h})...")

            upscaler = RealESRGANUpscaler(
                model_name=self.upscale_model,
                weights_dir=os.path.join(self.weights_dir, "realesrgan"),
                tile=effective_tile,
                half=True,
                device=self.device,
            )

            restorer = None
            is_outfit_only = ("Pakaian Saja" in self.focus_mode) or ("Outfit Only" in self.focus_mode)
            if not is_outfit_only:
                restorer = FaceRestorer(
                    method=self.face_restorer_type,
                    fidelity_weight=self.fidelity_weight,
                    scale=eff_scale,
                    weights_dir=self.weights_dir,
                    half=False,
                    device=self.device,
                )

            if restorer is not None and self.sequential_offload and self.device == "cuda":
                # Offload restorer to start with upscaler on GPU
                restorer.offload_to_cpu()

            # 3. Sequential Frame Enhancement Loop
            mid_idx = total_frames // 2
            preview_before = os.path.join(output_dir, f"{base_name}_before_preview.png")
            preview_after = os.path.join(output_dir, f"{base_name}_after_preview.png")

            enhanced_paths = []
            with torch.inference_mode():
                for i, frame_path in enumerate(frame_files):
                    img_bgr = cv2.imread(frame_path)
                    if img_bgr is None:
                        continue

                    if i == mid_idx:
                        cv2.imwrite(preview_before, img_bgr)

                    # Step A: Upscale background and outfit with Real-ESRGAN (tiled)
                    if self.sequential_offload and self.device == "cuda":
                        upscaler.reload_to_gpu()
                    upscaled_canvas = upscaler.enhance_frame(img_bgr, outscale=target_scale)

                    # Safe clamp to target resolution (prevents 5K/8K runaway RAM usage)
                    if upscaled_canvas.shape[1] != target_w or upscaled_canvas.shape[0] != target_h:
                        upscaled_canvas = cv2.resize(
                            upscaled_canvas,
                            (target_w, target_h),
                            interpolation=cv2.INTER_AREA if upscaled_canvas.shape[0] > target_h else cv2.INTER_LANCZOS4
                        )

                    # Step B: Restore Face if enabled, else use upscaled canvas directly
                    if restorer is not None:
                        if self.sequential_offload and self.device == "cuda":
                            upscaler.offload_to_cpu()
                            restorer.reload_to_gpu()

                        enhanced_frame = restorer.restore_face(img_bgr, upsampled_canvas=upscaled_canvas)

                        if self.sequential_offload and self.device == "cuda":
                            restorer.offload_to_cpu()
                    else:
                        enhanced_frame = upscaled_canvas

                    # Step C: Fabric Texture Sharpening & Color Vibrancy Boost for Outfits
                    if self.fabric_sharpness > 0.01:
                        blur = cv2.GaussianBlur(enhanced_frame, (0, 0), sigmaX=1.2, sigmaY=1.2)
                        amount = float(self.fabric_sharpness) * 0.75
                        enhanced_frame = cv2.addWeighted(enhanced_frame, 1.0 + amount, blur, -amount, 0)

                    if self.color_boost > 0.01:
                        hsv = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + float(self.color_boost) * 0.35), 0, 255)
                        enhanced_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                    # Save enhanced frame to disk immediately (free RAM)
                    out_frame_path = os.path.join(enhanced_frames_dir, f"frame_{i:06d}.png")
                    cv2.imwrite(out_frame_path, enhanced_frame)
                    enhanced_paths.append(out_frame_path)

                    if i == mid_idx:
                        cv2.imwrite(preview_after, enhanced_frame)

                    # Free frame memory and empty CUDA cache
                    del img_bgr, upscaled_canvas, enhanced_frame
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if i % 10 == 0:
                        gc.collect()

                    # Update progress
                    progress_val = 0.10 + (0.75 * (i + 1) / total_frames)
                    if progress_callback:
                        progress_callback(
                            progress_val,
                            f"Memproses frame {i + 1}/{total_frames} ({(i + 1) * 100 // total_frames}%)"
                        )

            # Cleanup models from VRAM
            del upscaler
            if restorer is not None:
                del restorer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # 4. Optional RIFE Frame Interpolation (2x FPS)
            final_frames_dir = enhanced_frames_dir
            final_fps = fps

            if self.enable_rife:
                rife_weight = os.path.join(self.weights_dir, "rife", "flownet.pkl")
                if os.path.exists(rife_weight):
                    if progress_callback:
                        progress_callback(0.86, "Smoothing motion with RIFE (2x FPS)...")
                    try:
                        from pipeline.interpolate import RIFEInterpolater
                        rife = RIFEInterpolater(weights_path=rife_weight, device=self.device, half=True)
                        interp_dir = os.path.join(work_dir, "interpolated_frames")
                        rife.interpolate_sequence(enhanced_paths, interp_dir)
                        final_frames_dir = interp_dir
                        final_fps = fps * 2.0
                        del rife
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"[WARN] RIFE interpolation skipped due to: {e}")
                else:
                    print(f"[WARN] RIFE weights not found at {rife_weight}, skipping.")

            # 5. Reassemble into final MP4 video
            if progress_callback:
                progress_callback(0.92, "Stitching video and multiplexing audio...")

            reassemble_video(
                frames_dir=final_frames_dir,
                output_video_path=output_video_path,
                fps=final_fps,
                audio_path=audio_path,
                frame_format="png",
                crf=18,
            )

            if progress_callback:
                progress_callback(1.0, "Enhancement completed successfully!")

            return output_video_path, preview_before, preview_after

        finally:
            # Safely cleanup temporary frames
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass
