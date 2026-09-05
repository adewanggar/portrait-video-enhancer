"""
app.py
Portrait Video Enhancer - Gradio Desktop Application.
Enhance human portrait videos offline on RTX 3050 6GB VRAM.
Supports English (default) with dynamic language switch to Bahasa Indonesia.
Default Gradio Orange Theme, Clean UI, and Cancel button support.

Author: adewanggar
License: MIT
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch
import gradio as gr
from pipeline.enhancer import VideoEnhancer
from download_models import verify_models

HAS_CUDA = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else "CPU"

I18N = {
    "English": {
        "title": "Portrait Video Enhancer",
        "header_md": (
            "# Portrait Video Enhancer\n"
            "**Human Portrait Video Enhancer (Face & Outfit)**  \n"
            "Active Device: {gpu} | VRAM Optimization: 6GB FP16"
        ),
        "lang_label": "🌐 Language / Bahasa",
        "sec_upload": "### 1. Upload Video",
        "input_video_label": "Portrait Video (MP4/MOV/AVI)",
        "sec_settings": "### 2. Enhancement Settings",
        "focus_label": "Enhancement Focus",
        "focus_choices": [
            "Outfit + Face (Full)",
            "Outfit Only (Super Fast ~1-2 Min)",
        ],
        "focus_info": "Choose 'Outfit Only' for fashion / UGC clothing videos (much faster)",
        "fabric_label": "Fabric & Texture Sharpness (0.0 - 1.0)",
        "fabric_info": "Enhances fabric grain, knitwear, embroidery, and clothing folds to appear crisp & vivid",
        "color_label": "Clothing Color Vibrance & Brightness (0.0 - 1.0)",
        "color_info": "Boosts outfit color saturation for an attractive catalog look",
        "restorer_label": "Face Restoration",
        "restorer_info": "CodeFormer (Original identity) or GFPGAN (Smooth skin)",
        "upscale_label": "Outfit & Background Upscaler",
        "upscale_info": "x2plus (4x faster, ideal for 360p) or x4plus (maximum detail)",
        "target_res_label": "Target Output Resolution",
        "target_res_choices": [
            "1080p Full HD (1080x1920)",
            "720p HD (720x1280)",
            "2K QHD (1440x2560)",
            "4K UHD (2160x3840)",
            "Original Model Scale (2x/4x)",
        ],
        "target_res_info": "1080p Full HD is optimal for TikTok/Reels & VRAM efficient",
        "fidelity_label": "CodeFormer Facial Fidelity Weight (0.0 - 1.0)",
        "fidelity_info": "0.0 = high generative detail | 1.0 = exact original face (0.60 - 0.70 most natural)",
        "mode_label": "Processing Mode",
        "mode_choices": ["Fast mode", "Quality mode"],
        "mode_info": "Fast mode: untiled for <=720p | Quality mode: precise tiling",
        "rife_label": "Smooth Motion (RIFE 2x FPS)",
        "rife_info": "Frame interpolation for 60 fps",
        "process_btn": "Enhance Video",
        "cancel_btn": "Cancel",
        "status_ready": "Hardware: {gpu} | Status: Ready (Offline)",
        "sec_output": "### 3. Video Result",
        "output_video_label": "Enhanced Video Result",
        "download_file_label": "Download MP4 File",
        "sec_preview": "### 4. Middle Frame Comparison",
        "preview_before_label": "Before (Original)",
        "preview_after_label": "After (Enhanced)",
        "acc_model_status": "Model Weights Status",
        "model_status_label": "File Status",
        "model_status_init": "Click the button below to verify model weights in local storage.",
        "check_btn": "Check Model Weights Status",
        "err_no_video": "Please select or upload a video file first.",
        "err_not_found": "Video file not found: {path}",
        "process_success": "Video processing completed successfully.",
        "process_failed": "Process failed: {error}",
        "models_ok": "All model weights are complete and ready to use.",
        "models_missing": "Some model weights are missing. Run: python download_models.py",
        "progress_init": "Preparing video enhancement...",
    },
    "Bahasa Indonesia": {
        "title": "Portrait Video Enhancer",
        "header_md": (
            "# Portrait Video Enhancer\n"
            "**Peningkat Kualitas Video Portrait Manusia (Wajah & Pakaian)**  \n"
            "Perangkat Aktif: {gpu} | Optimasi VRAM: 6GB FP16"
        ),
        "lang_label": "🌐 Language / Bahasa",
        "sec_upload": "### 1. Unggah Video",
        "input_video_label": "Video Portrait (MP4/MOV/AVI)",
        "sec_settings": "### 2. Pengaturan Peningkatan",
        "focus_label": "Fokus Peningkatan",
        "focus_choices": [
            "Pakaian + Wajah (Lengkap)",
            "Fokus Pakaian Saja (Super Cepat ~1-2 Menit)",
        ],
        "focus_info": "Pilih 'Fokus Pakaian Saja' untuk video UGC fashion / pakaian (jauh lebih cepat)",
        "fabric_label": "Ketajaman Serat & Tekstur Kain (0.0 - 1.0)",
        "fabric_info": "Mempertegas serat kain, rajutan, sulaman, dan lipatan baju agar timbul & jelas",
        "color_label": "Vibransi & Kecerahan Warna Baju (0.0 - 1.0)",
        "color_info": "Meningkatkan saturasi warna pakaian agar memikat pembeli (khas katalog)",
        "restorer_label": "Restorasi Wajah",
        "restorer_info": "CodeFormer (Identitas asli) atau GFPGAN (Kulit mulus)",
        "upscale_label": "Upscaler Pakaian & Latar",
        "upscale_info": "x2plus (4x lebih cepat, ideal 360p) atau x4plus (detail maksimal)",
        "target_res_label": "Resolusi Output Target",
        "target_res_choices": [
            "1080p Full HD (1080x1920)",
            "720p HD (720x1280)",
            "2K QHD (1440x2560)",
            "4K UHD (2160x3840)",
            "Skala Asli Model (2x/4x)",
        ],
        "target_res_info": "1080p Full HD adalah standar terbaik TikTok/Reels & hemat RAM",
        "fidelity_label": "Bobot Kemiripan Wajah CodeFormer (0.0 - 1.0)",
        "fidelity_info": "0.0 = detail generatif tinggi | 1.0 = persis wajah asli (0.60 - 0.70 paling natural)",
        "mode_label": "Mode Pemrosesan",
        "mode_choices": ["Fast mode", "Quality mode"],
        "mode_info": "Fast mode: tanpa tiling untuk <=720p | Quality mode: tiling presisi",
        "rife_label": "Gerakan Halus (RIFE 2x FPS)",
        "rife_info": "Interpolasi frame untuk 60 fps",
        "process_btn": "Proses Video",
        "cancel_btn": "Batal",
        "status_ready": "Hardware: {gpu} | Status: Siap Offline",
        "sec_output": "### 3. Hasil Video",
        "output_video_label": "Video Hasil Peningkatan",
        "download_file_label": "Unduh Berkas MP4",
        "sec_preview": "### 4. Perbandingan Frame Tengah",
        "preview_before_label": "Sebelum (Asli)",
        "preview_after_label": "Sesudah (Peningkatan)",
        "acc_model_status": "Status Bobot Model",
        "model_status_label": "Status Berkas",
        "model_status_init": "Klik tombol di bawah untuk memeriksa bobot model di penyimpanan lokal.",
        "check_btn": "Periksa Status Bobot Model",
        "err_no_video": "Silakan pilih atau unggah berkas video terlebih dahulu.",
        "err_not_found": "Berkas video tidak ditemukan: {path}",
        "process_success": "Proses video selesai dengan sukses.",
        "process_failed": "Proses gagal: {error}",
        "models_ok": "Semua bobot model lengkap dan siap digunakan.",
        "models_missing": "Sebagian bobot model belum terunduh. Jalankan: python download_models.py",
        "progress_init": "Mempersiapkan peningkatan video...",
    },
}

CUSTOM_CSS = """
#header-row {
    align-items: center;
    border-bottom: 1px solid rgba(255, 140, 0, 0.25);
    padding-bottom: 12px;
    margin-bottom: 16px;
}
#lang-switch-box {
    display: flex;
    justify-content: flex-end;
    align-items: flex-start;
}
#lang-switch-box .gradio-radio {
    background: rgba(255, 140, 0, 0.06);
    border: 1px solid rgba(255, 140, 0, 0.35);
    border-radius: 8px;
    padding: 6px 12px;
}
"""


def process_video_ui(
    video_file,
    restorer_choice,
    fidelity,
    upscale_choice,
    target_res,
    focus_choice,
    fabric_sharpness,
    color_boost,
    rife_toggle,
    run_mode,
    language="English",
    progress=gr.Progress(track_tqdm=True),
):
    t = I18N.get(language, I18N["English"])

    if video_file is None:
        raise gr.Error(t["err_no_video"])

    input_path = video_file
    if isinstance(video_file, dict) and "name" in video_file:
        input_path = video_file["name"]

    if not os.path.exists(input_path):
        raise gr.Error(t["err_not_found"].format(path=input_path))

    tile_size = 400 if run_mode == "Fast mode" else 300

    enhancer = VideoEnhancer(
        face_restorer_type=restorer_choice,
        fidelity_weight=float(fidelity),
        upscale_model=upscale_choice,
        tile_size=tile_size,
        enable_rife=bool(rife_toggle),
        mode=run_mode,
        sequential_offload=False,
        weights_dir="weights",
        target_resolution=target_res,
        focus_mode=focus_choice,
        fabric_sharpness=float(fabric_sharpness),
        color_boost=float(color_boost),
    )

    def ui_progress(pct, msg):
        progress(pct, desc=msg)

    try:
        out_video, before_img, after_img = enhancer.process_video(
            input_video_path=input_path,
            progress_callback=ui_progress,
        )
        return out_video, before_img, after_img, out_video, t["process_success"]
    except Exception as e:
        raise gr.Error(t["process_failed"].format(error=str(e)))


def check_models_status(language="English"):
    t = I18N.get(language, I18N["English"])
    ok = verify_models("weights")
    if ok:
        return t["models_ok"]
    return t["models_missing"]


def on_language_change(lang, curr_focus, curr_target_res):
    t = I18N.get(lang, I18N["English"])
    gpu_label = GPU_NAME if HAS_CUDA else ("CPU (CUDA not available)" if lang == "English" else "CPU (CUDA tidak tersedia)")

    # Map current focus selection to new language
    if curr_focus and ("Outfit Only" in curr_focus or "Pakaian Saja" in curr_focus):
        new_focus_val = t["focus_choices"][1]
    else:
        new_focus_val = t["focus_choices"][0]

    # Map current target resolution to new language
    new_res_val = t["target_res_choices"][0]
    if curr_target_res:
        if "720p" in curr_target_res:
            new_res_val = t["target_res_choices"][1]
        elif "2K" in curr_target_res:
            new_res_val = t["target_res_choices"][2]
        elif "4K" in curr_target_res:
            new_res_val = t["target_res_choices"][3]
        elif "Original" in curr_target_res or "Asli" in curr_target_res:
            new_res_val = t["target_res_choices"][4]
        else:
            new_res_val = t["target_res_choices"][0]

    return [
        gr.update(value=t["header_md"].format(gpu=gpu_label)),
        gr.update(value=t["sec_upload"]),
        gr.update(label=t["input_video_label"]),
        gr.update(value=t["sec_settings"]),
        gr.update(
            label=t["focus_label"],
            choices=t["focus_choices"],
            value=new_focus_val,
            info=t["focus_info"],
        ),
        gr.update(
            label=t["fabric_label"],
            info=t["fabric_info"],
        ),
        gr.update(
            label=t["color_label"],
            info=t["color_info"],
        ),
        gr.update(
            label=t["restorer_label"],
            info=t["restorer_info"],
        ),
        gr.update(
            label=t["upscale_label"],
            info=t["upscale_info"],
        ),
        gr.update(
            label=t["target_res_label"],
            choices=t["target_res_choices"],
            value=new_res_val,
            info=t["target_res_info"],
        ),
        gr.update(
            label=t["fidelity_label"],
            info=t["fidelity_info"],
        ),
        gr.update(
            label=t["mode_label"],
            choices=t["mode_choices"],
            info=t["mode_info"],
        ),
        gr.update(
            label=t["rife_label"],
            info=t["rife_info"],
        ),
        gr.update(value=t["process_btn"]),
        gr.update(value=t["cancel_btn"]),
        gr.update(value=t["status_ready"].format(gpu=gpu_label)),
        gr.update(value=t["sec_output"]),
        gr.update(label=t["output_video_label"]),
        gr.update(label=t["download_file_label"]),
        gr.update(value=t["sec_preview"]),
        gr.update(label=t["preview_before_label"]),
        gr.update(label=t["preview_after_label"]),
        gr.update(label=t["acc_model_status"]),
        gr.update(label=t["model_status_label"]),
        gr.update(value=t["check_btn"]),
    ]


default_lang = "English"
t_def = I18N[default_lang]
gpu_initial = GPU_NAME if HAS_CUDA else "CPU (CUDA not available)"

with gr.Blocks(title=t_def["title"], css=CUSTOM_CSS) as demo:
    with gr.Column():
        with gr.Row(elem_id="header-row"):
            with gr.Column(scale=8):
                header_md = gr.Markdown(
                    t_def["header_md"].format(gpu=gpu_initial)
                )
            with gr.Column(scale=3, min_width=220, elem_id="lang-switch-box"):
                lang_switch = gr.Radio(
                    choices=["English", "Bahasa Indonesia"],
                    value=default_lang,
                    label=t_def["lang_label"],
                    interactive=True,
                )

        with gr.Row():
            # Left Column: Input & Settings
            with gr.Column(scale=5):
                sec_upload_md = gr.Markdown(t_def["sec_upload"])
                input_video = gr.Video(
                    label=t_def["input_video_label"],
                    sources=["upload"],
                    height=280,
                )

                sec_settings_md = gr.Markdown(t_def["sec_settings"])
                focus_radio = gr.Radio(
                    label=t_def["focus_label"],
                    choices=t_def["focus_choices"],
                    value=t_def["focus_choices"][0],
                    info=t_def["focus_info"],
                )

                with gr.Row():
                    fabric_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.35,
                        step=0.05,
                        label=t_def["fabric_label"],
                        info=t_def["fabric_info"],
                    )
                    color_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.15,
                        step=0.05,
                        label=t_def["color_label"],
                        info=t_def["color_info"],
                    )

                with gr.Row():
                    restorer_dropdown = gr.Dropdown(
                        label=t_def["restorer_label"],
                        choices=["CodeFormer", "GFPGAN"],
                        value="CodeFormer",
                        info=t_def["restorer_info"],
                    )
                    upscale_dropdown = gr.Dropdown(
                        label=t_def["upscale_label"],
                        choices=["RealESRGAN_x2plus", "RealESRGAN_x4plus"],
                        value="RealESRGAN_x2plus",
                        info=t_def["upscale_info"],
                    )

                target_res_dropdown = gr.Dropdown(
                    label=t_def["target_res_label"],
                    choices=t_def["target_res_choices"],
                    value=t_def["target_res_choices"][0],
                    info=t_def["target_res_info"],
                )

                fidelity_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.65,
                    step=0.05,
                    label=t_def["fidelity_label"],
                    info=t_def["fidelity_info"],
                )

                with gr.Row():
                    mode_radio = gr.Radio(
                        label=t_def["mode_label"],
                        choices=t_def["mode_choices"],
                        value=t_def["mode_choices"][0],
                        info=t_def["mode_info"],
                    )
                    rife_checkbox = gr.Checkbox(
                        label=t_def["rife_label"],
                        value=False,
                        info=t_def["rife_info"],
                    )

                with gr.Row():
                    process_btn = gr.Button(
                        t_def["process_btn"],
                        variant="primary",
                        scale=3,
                    )
                    cancel_btn = gr.Button(
                        t_def["cancel_btn"],
                        variant="stop",
                        scale=1,
                    )

                status_text = gr.Markdown(
                    t_def["status_ready"].format(gpu=gpu_initial)
                )

            # Right Column: Output & Preview
            with gr.Column(scale=6):
                sec_output_md = gr.Markdown(t_def["sec_output"])
                output_video = gr.Video(
                    label=t_def["output_video_label"],
                    interactive=False,
                    height=320,
                )

                download_file = gr.File(
                    label=t_def["download_file_label"],
                    interactive=False,
                )

                sec_preview_md = gr.Markdown(t_def["sec_preview"])
                with gr.Row():
                    preview_before_img = gr.Image(
                        label=t_def["preview_before_label"],
                        type="filepath",
                        height=220,
                    )
                    preview_after_img = gr.Image(
                        label=t_def["preview_after_label"],
                        type="filepath",
                        height=220,
                    )

        with gr.Accordion(t_def["acc_model_status"], open=False) as model_status_accordion:
            model_status_box = gr.Textbox(
                label=t_def["model_status_label"],
                value=t_def["model_status_init"],
                interactive=False,
            )
            check_btn = gr.Button(t_def["check_btn"])
            check_btn.click(
                fn=check_models_status,
                inputs=[lang_switch],
                outputs=model_status_box,
            )

        # Wire up language switch in top right corner
        lang_switch.change(
            fn=on_language_change,
            inputs=[lang_switch, focus_radio, target_res_dropdown],
            outputs=[
                header_md,
                sec_upload_md,
                input_video,
                sec_settings_md,
                focus_radio,
                fabric_slider,
                color_slider,
                restorer_dropdown,
                upscale_dropdown,
                target_res_dropdown,
                fidelity_slider,
                mode_radio,
                rife_checkbox,
                process_btn,
                cancel_btn,
                status_text,
                sec_output_md,
                output_video,
                download_file,
                sec_preview_md,
                preview_before_img,
                preview_after_img,
                model_status_accordion,
                model_status_box,
                check_btn,
            ],
        )

        # Wire up execution and cancel button
        process_event = process_btn.click(
            fn=process_video_ui,
            inputs=[
                input_video,
                restorer_dropdown,
                fidelity_slider,
                upscale_dropdown,
                target_res_dropdown,
                focus_radio,
                fabric_slider,
                color_slider,
                rife_checkbox,
                mode_radio,
                lang_switch,
            ],
            outputs=[
                output_video,
                preview_before_img,
                preview_after_img,
                download_file,
                status_text,
            ],
        )

        cancel_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[process_event],
        )

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Default(primary_hue="orange"),
    )
