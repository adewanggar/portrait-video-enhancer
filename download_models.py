#!/usr/bin/env python3
"""
download_models.py
Auto-downloader for Portrait Video Enhancer weights.
Downloads CodeFormer, facelib, Real-ESRGAN, GFPGAN, and RIFE weights.
"""

import os
import sys
import argparse
import urllib.request

MODELS = {
    "codeformer": [
        {
            "name": "codeformer.pth",
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            "dest": os.path.join("weights", "CodeFormer", "codeformer.pth"),
            "size_approx_mb": 377,
        },
        {
            "name": "detection_Resnet50_Final.pth",
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            "dest": os.path.join("weights", "facelib", "detection_Resnet50_Final.pth"),
            "size_approx_mb": 109,
        },
        {
            "name": "parsing_parsenet.pth",
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
            "dest": os.path.join("weights", "facelib", "parsing_parsenet.pth"),
            "size_approx_mb": 85,
        },
    ],
    "realesrgan": [
        {
            "name": "RealESRGAN_x4plus.pth",
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "dest": os.path.join("weights", "realesrgan", "RealESRGAN_x4plus.pth"),
            "size_approx_mb": 67,
        },
        {
            "name": "RealESRGAN_x2plus.pth",
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "dest": os.path.join("weights", "realesrgan", "RealESRGAN_x2plus.pth"),
            "size_approx_mb": 67,
        },
    ],
    "gfpgan": [
        {
            "name": "GFPGANv1.4.pth",
            "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            "dest": os.path.join("weights", "gfpgan", "GFPGANv1.4.pth"),
            "size_approx_mb": 348,
        }
    ],
    "rife": [
        {
            "name": "flownet.pkl",
            "url": "https://huggingface.co/Upsampler/rife-4-25/resolve/main/flownet.pkl",
            "dest": os.path.join("weights", "rife", "flownet.pkl"),
            "size_approx_mb": 170,
        }
    ],
}


def download_file(url: str, dest_path: str, name: str) -> bool:
    """Download a file with progress display."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024 * 1024:
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"[OK] {name} already exists ({size_mb:.1f} MB): {dest_path}")
        return True

    print(f"[DOWNLOADING] {name} from {url} ...")
    temp_dest = dest_path + ".download"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        with urllib.request.urlopen(req) as response, open(temp_dest, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 512  # 512 KB

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)

                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    mb_curr = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(
                        f"\r -> {name}: {percent:5.1f}% [{mb_curr:6.1f}MB / {mb_total:6.1f}MB]",
                        end="",
                        flush=True,
                    )
                else:
                    mb_curr = downloaded / (1024 * 1024)
                    print(f"\r -> {name}: {mb_curr:6.1f} MB downloaded", end="", flush=True)

            print()

        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_dest, dest_path)
        print(f"[DONE] Saved {name} -> {dest_path}")
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to download {name}: {e}")
        if os.path.exists(temp_dest):
            try:
                os.remove(temp_dest)
            except OSError:
                pass
        return False


def verify_models(base_dir: str = "."):
    """Check which models are present on disk."""
    print("\n--- Weight Verification Status ---")
    all_present = True
    for category, items in MODELS.items():
        print(f"\n[{category.upper()}]")
        for item in items:
            full_path = os.path.join(base_dir, item["dest"])
            if os.path.exists(full_path) and os.path.getsize(full_path) > 1024:
                sz = os.path.getsize(full_path) / (1024 * 1024)
                print(f"  [OK]      {item['name']:<30} ({sz:.1f} MB) -> {item['dest']}")
            else:
                print(f"  [MISSING] {item['name']:<30} -> {item['dest']}")
                all_present = False
    return all_present


def main():
    parser = argparse.ArgumentParser(description="Download model weights for Portrait Video Enhancer")
    parser.add_argument("--all", action="store_true", help="Download all model weights")
    parser.add_argument("--category", choices=["codeformer", "realesrgan", "gfpgan", "rife"], help="Download specific category")
    parser.add_argument("--verify", action="store_true", help="Verify presence of model weights without downloading")
    args = parser.parse_args()

    if args.verify:
        verify_models()
        return

    categories = [args.category] if args.category else MODELS.keys()
    success = True
    for cat in categories:
        print(f"\n=== Processing Category: {cat} ===")
        for item in MODELS[cat]:
            ok = download_file(item["url"], item["dest"], item["name"])
            if not ok:
                success = False

    verify_models()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
