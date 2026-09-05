"""
pipeline/reassemble.py
Reassemble enhanced frames and original audio track back into MP4 video using FFmpeg.
"""

import os
import subprocess
from typing import Optional


def reassemble_video(
    frames_dir: str,
    output_video_path: str,
    fps: float = 30.0,
    audio_path: Optional[str] = None,
    frame_format: str = "png",
    crf: int = 18,
) -> str:
    """
    Stitch frame images from frames_dir into an MP4 video and mux original audio.
    """
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    frame_pattern = os.path.join(frames_dir, f"frame_%06d.{frame_format}")

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
    ]

    if audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        cmd.extend(["-i", audio_path])
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_video_path,
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            output_video_path,
        ])

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg reassembly failed: {res.stderr}")

    return output_video_path
