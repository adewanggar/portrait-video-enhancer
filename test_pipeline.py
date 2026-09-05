"""
test_pipeline.py
Quick self-contained verification of video extraction, probe, and reassembly.
Runs with standard ffmpeg on Windows.
"""

import os
import subprocess
import tempfile
import shutil
from pipeline.extract import probe_video, extract_frames_and_audio
from pipeline.reassemble import reassemble_video


def run_self_test():
    test_dir = tempfile.mkdtemp(prefix="enhancer_test_")
    sample_mp4 = os.path.join(test_dir, "sample.mp4")

    print("[TEST] Generating 1-second synthetic test video with FFmpeg...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
        "-c:v", "libx264",
        "-c:a", "aac",
        sample_mp4,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert os.path.exists(sample_mp4), "Sample video generation failed."
    print(f"  [OK] Generated test video: {sample_mp4}")

    # 1. Test probe
    info = probe_video(sample_mp4)
    print(f"  [OK] Probed metadata: {info['width']}x{info['height']} @ {info['fps']}fps, audio={info['has_audio']}")
    assert info["width"] == 320 and info["height"] == 240, "Incorrect probed dimensions"
    assert info["has_audio"] is True, "Audio stream not detected"

    # 2. Test frame & audio extraction
    extract_dir = os.path.join(test_dir, "extracted")
    meta = extract_frames_and_audio(sample_mp4, extract_dir)
    print(f"  [OK] Extracted {meta['total_frames']} frames to {meta['frames_dir']}")
    assert meta["total_frames"] == 10, f"Expected 10 frames, got {meta['total_frames']}"
    assert meta["audio_path"] is not None and os.path.exists(meta["audio_path"]), "Audio extraction failed"

    # 3. Test video reassembly
    out_mp4 = os.path.join(test_dir, "reassembled.mp4")
    reassemble_video(
        frames_dir=meta["frames_dir"],
        output_video_path=out_mp4,
        fps=meta["fps"],
        audio_path=meta["audio_path"],
        frame_format="png",
        crf=18,
    )
    assert os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 1000, "Reassembled video missing or empty"
    print(f"  [OK] Successfully reassembled video with audio: {out_mp4} ({os.path.getsize(out_mp4)} bytes)")

    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)
    print("\n[PASSED] All pipeline extract and reassembly tests passed successfully!")


if __name__ == "__main__":
    run_self_test()
