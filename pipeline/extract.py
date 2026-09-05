"""
pipeline/extract.py
Extract frames, audio track, and metadata (fps, resolution, frame count) from input video using FFmpeg.
"""

import os
import re
import json
import subprocess
from typing import Dict, Any, Optional


def probe_video(video_path: str) -> Dict[str, Any]:
    """
    Probe video file to extract fps, dimensions, duration, and audio presence.
    Uses ffprobe if available, with regex fallback from ffmpeg -i.
    """
    info = {
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "duration": 0.0,
        "total_frames": 0,
        "has_audio": False,
    }

    # Try ffprobe with JSON output
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                info["width"] = int(stream.get("width", 1920))
                info["height"] = int(stream.get("height", 1080))
                
                # Parse r_frame_rate (e.g., "30/1" or "30000/1001")
                r_fps = stream.get("r_frame_rate", "30/1")
                if "/" in r_fps:
                    num, den = map(float, r_fps.split("/"))
                    info["fps"] = num / den if den != 0 else 30.0
                else:
                    info["fps"] = float(r_fps)

                # Total frames if explicitly reported
                nb_frames = stream.get("nb_frames")
                if nb_frames and nb_frames.isdigit():
                    info["total_frames"] = int(nb_frames)

            elif stream.get("codec_type") == "audio":
                info["has_audio"] = True

        duration = data.get("format", {}).get("duration")
        if duration:
            info["duration"] = float(duration)
            if info["total_frames"] == 0 and info["fps"] > 0:
                info["total_frames"] = max(1, int(info["duration"] * info["fps"]))

        return info

    except Exception:
        pass

    # Fallback to ffmpeg -i inspection
    try:
        cmd = ["ffmpeg", "-i", video_path]
        res = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        output = res.stderr

        if "Audio:" in output:
            info["has_audio"] = True

        fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", output)
        if fps_match:
            info["fps"] = float(fps_match.group(1))

        dim_match = re.search(r"(\d{3,4})x(\d{3,4})", output)
        if dim_match:
            info["width"] = int(dim_match.group(1))
            info["height"] = int(dim_match.group(2))

        dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", output)
        if dur_match:
            h, m, s = map(float, dur_match.groups())
            info["duration"] = h * 3600 + m * 60 + s
            info["total_frames"] = max(1, int(info["duration"] * info["fps"]))

    except Exception as e:
        print(f"[WARN] Error probing video {video_path}: {e}")

    return info


def extract_frames_and_audio(
    video_path: str,
    output_dir: str,
    frame_format: str = "png"
) -> Dict[str, Any]:
    """
    Extract video frames into PNG files and save the audio track.
    Returns dictionary with extracted metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "input_frames")
    os.makedirs(frames_dir, exist_ok=True)

    info = probe_video(video_path)

    # 1. Extract frames sequentially using ffmpeg
    frame_pattern = os.path.join(frames_dir, f"frame_%06d.{frame_format}")
    extract_cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-qscale:v", "1",
        "-qmin", "1",
        frame_pattern,
    ]
    subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Count extracted frames on disk
    extracted_frames = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(f".{frame_format}")]
    )
    info["total_frames"] = len(extracted_frames)
    info["frame_files"] = extracted_frames
    info["frames_dir"] = frames_dir

    # 2. Extract audio if present
    audio_path = None
    if info["has_audio"]:
        audio_dest = os.path.join(output_dir, "extracted_audio.aac")
        # Try stream copy first
        copy_cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-c:a", "copy",
            audio_dest,
        ]
        res = subprocess.run(copy_cmd, capture_output=True)
        if res.returncode == 0 and os.path.exists(audio_dest) and os.path.getsize(audio_dest) > 0:
            audio_path = audio_dest
        else:
            # Fallback to re-encoding as aac
            aac_dest = os.path.join(output_dir, "extracted_audio.aac")
            re_cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-vn",
                "-c:a", "aac",
                "-b:a", "192k",
                aac_dest,
            ]
            res2 = subprocess.run(re_cmd, capture_output=True)
            if res2.returncode == 0 and os.path.exists(aac_dest) and os.path.getsize(aac_dest) > 0:
                audio_path = aac_dest

    info["audio_path"] = audio_path
    return info
