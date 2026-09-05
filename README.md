# Portrait Video Enhancer (RTX 3050 6GB VRAM)

<div align="center">

**Choose Language / Pilih Bahasa:**  
[🇬🇧 English](#english) &bull; [🇮🇩 Bahasa Indonesia](#bahasa-indonesia)

<br>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adewanggar/portrait-video-enhancer/blob/main/portrait_video_enhancer_colab.ipynb)

</div>

---

<a name="english"></a>
# 🇬🇧 English Documentation

A 100% offline, local desktop application tailored for enhancing **human portrait videos** (facial fidelity, skin details, clothing textures, and outfits) using deep learning. Designed from the ground up to operate reliably within the **6GB VRAM** ceiling of the NVIDIA GeForce RTX 3050 Laptop GPU.

---

## Key Features

- **Sequential Pipeline**: Processes one frame at a time without accumulating uncompressed frames in system RAM.
- **Tiled Background & Outfit Upscaling**: Employs **Real-ESRGAN** (`RealESRGAN_x4plus` / `x2plus`) with tile-based inference (`tile=400` / `300`) to prevent Out-Of-Memory (OOM) errors.
- **High-Fidelity Face Restoration**: Uses **CodeFormer** with adjustable `fidelity_weight` (0.0 - 1.0) for natural facial identity preservation, with **GFPGAN** available as a fully permissive commercial alternative.
- **Sequential GPU Memory Offloading**: Alternates model residence on CUDA so heavy models never congest VRAM simultaneously.
- **FP16 Half-Precision**: Native half-precision inference for minimal VRAM footprint and maximum RTX tensor throughput.
- **Motion Smoothing (RIFE)**: Optional 2x frame rate interpolation using Practical-RIFE for fluid motion (e.g., 30 fps to 60 fps).
- **Gradio Local Desktop UI with Dynamic Language Switch**:
  - **English Default**: Clean, modern interface configured in English by default.
  - **Top-Right Language Switch**: Interactive switch button in the top-right corner to toggle instantly between **English** and **Bahasa Indonesia**.
  - **Real-Time UI Update**: All labels, sliders, tooltips, dropdowns, and buttons update reactively upon switching.
  - **Drag-and-Drop Video Upload**: Per-frame real-time progress bar, middle-frame before/after preview, and one-click MP4 download.

---

## User Interface & Language Switch

The application opens in **English** by default. At any time, you can switch languages using the selector in the top-right corner:

1. Locate the **🌐 Language / Bahasa** switch in the top-right corner of the header.
2. Select **Bahasa Indonesia** to switch all labels, tooltips, dropdown options, and status messages to Indonesian.
3. Select **English** to switch back to English.
4. Your active settings and parameters are automatically preserved when switching languages.

---

## Project Structure

```
portrait-video-enhancer/
├── app.py                  # Entry point, Gradio UI with bilingual switch
├── pipeline/
│   ├── __init__.py         # Runtime compatibility patches (torchvision/basicsr)
│   ├── extract.py          # FFmpeg frame extraction + audio + metadata probe
│   ├── face_restore.py     # CodeFormer & GFPGAN wrapper with VRAM offload
│   ├── upscale.py          # Real-ESRGAN tiled fp16 wrapper
│   ├── interpolate.py      # RIFE motion smoothing (2x FPS)
│   ├── enhancer.py         # Master frame-by-frame enhancement coordinator
│   └── reassemble.py       # FFmpeg video + audio multiplexer
├── weights/                # Model weights directory
│   ├── CodeFormer/
│   ├── facelib/
│   ├── realesrgan/
│   ├── gfpgan/
│   └── rife/
├── fix_basicsr.py          # Automatic compatibility patch for BasicSR on PyTorch 2.x
├── download_models.py      # Automated weights downloader with progress & resume
├── test_pipeline.py        # Self-contained extraction and reassembly test
├── portrait_video_enhancer_colab.ipynb # Ready-to-run Google Colab Notebook (Free T4 GPU)
├── requirements.txt        # Minimal dependency list
├── run.bat                 # One-click launcher (sets up environment & port)
├── start.bat               # Shortcut launcher script
└── README.md               # Bilingual documentation (English & Bahasa Indonesia)
```

---

## Quickstart & Installation

### ⚡ Run on Google Colab (Free Cloud GPU)
If you don't have an RTX GPU locally, run the application directly on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adewanggar/portrait-video-enhancer/blob/main/portrait_video_enhancer_colab.ipynb)

1. Open the notebook via the button above.
2. Select **Runtime** > **Change runtime type** > **T4 GPU** (free tier).
3. Run all cells sequentially.
4. Open the public `*.gradio.live` link generated in the final cell.

---

### Local Installation

### 1. Requirements
- **OS**: Windows 10/11
- **Python**: 3.10 or 3.11
- **GPU**: NVIDIA RTX 3050 Laptop GPU (6GB VRAM) or higher
- **FFmpeg**: Installed and accessible in system `PATH` (verified with FFmpeg 9.0+)

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Apply BasicSR / PyTorch Compatibility Patch
BasicSR 1.4.2 has a known registry conflict on PyTorch 2.x. Run the one-step patch script:
```powershell
python fix_basicsr.py
```

### 4. Download Model Weights
Download all model weights automatically from official release repositories:
```powershell
python download_models.py --all
```

To download only specific categories:
```powershell
python download_models.py --category codeformer
python download_models.py --category realesrgan
python download_models.py --category gfpgan
python download_models.py --category rife
```

To verify weights already downloaded:
```powershell
python download_models.py --verify
```

### 5. Launch Application
You can launch the application with either:
```powershell
run.bat
```
or directly via Python:
```powershell
python app.py
```
Your default web browser will open automatically at: `http://127.0.0.1:7860`.

---

## 6GB VRAM Optimization Guidelines

| Feature | Fast Mode | Quality Mode |
| :--- | :--- | :--- |
| **Real-ESRGAN Tile Size** | 400 | 300 |
| **RIFE Motion Interpolation** | Off | Optional (2x FPS) |
| **Precision** | FP16 Half | FP16 Half |
| **Sequential VRAM Offload** | Active | Active |
| **Per-Frame Garbage Collection** | `torch.cuda.empty_cache()` | `torch.cuda.empty_cache()` |

### VRAM Budget Breakdown on RTX 3050 (6,144 MB)
- **Windows OS + Desktop Display**: ~400 MB
- **Real-ESRGAN (FP16, Tile 400)**: ~2,100 MB
- **CodeFormer + RetinaFace (FP16)**: ~1,800 MB
- **Safety Headroom**: >1,500 MB (guaranteed zero OOM)

---

## Author & License

- **Author**: adewanggar
- **License**: [MIT License](LICENSE)

### Third-Party Model Licenses & Commercial Usage Disclaimer

> [!IMPORTANT]
> This application incorporates multiple open-source models, each governed by its own respective license.

- **CodeFormer**: [NTU S-Lab License 1.0](https://github.com/sczhou/CodeFormer/blob/master/LICENSE) — **Strictly for Non-Commercial Research Use**. If you plan to use this application for commercial purposes, switch the face restorer dropdown to **GFPGAN**.
- **GFPGAN**: [Apache License 2.0](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE) — Permissive commercial and open-source usage allowed.
- **Real-ESRGAN**: [BSD 3-Clause License](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) — Free for commercial and private use with attribution.
- **RIFE (Practical-RIFE)**: [MIT License](https://github.com/hzwer/Practical-RIFE/blob/main/LICENSE) — Standard permissive open-source license.

---

<br>
<hr style="height:2px;border:none;color:#333;background-color:#333;" />
<br>

<a name="bahasa-indonesia"></a>
# 🇮🇩 Dokumentasi Bahasa Indonesia

Aplikasi desktop lokal 100% offline yang dirancang khusus untuk meningkatkan kualitas **video portrait manusia** (kejernihan wajah, detail kulit, serat kain pakaian, dan busana) menggunakan deep learning. Dioptimalkan dari awal agar berjalan stabil dalam batas kapasitas **6GB VRAM** NVIDIA GeForce RTX 3050 Laptop GPU.

---

## Fitur Utama

- **Pipeline Sekuensial**: Memproses frame satu per satu tanpa menumpuk frame mentah di memori RAM sistem.
- **Upscaling Pakaian & Latar Tiled**: Menggunakan **Real-ESRGAN** (`RealESRGAN_x4plus` / `x2plus`) dengan metode inferensi berbasis tile (`tile=400` / `300`) untuk mencegah error kehabisan memori (Out-Of-Memory / OOM).
- **Restorasi Wajah Fidelitas Tinggi**: Menggunakan **CodeFormer** dengan bobot kemiripan yang dapat disesuaikan (`fidelity_weight` 0.0 - 1.0) untuk menjaga identitas asli wajah, serta **GFPGAN** sebagai alternatif berlisensi komersial.
- **Offload Memori GPU Sekuensial**: Mengatur penempatan model di CUDA secara bergantian sehingga dua model besar tidak membebani VRAM secara bersamaan.
- **Presisi Setengah FP16**: Inferensi native half-precision untuk efisiensi VRAM maksimal dan throughput tensor core RTX yang tinggi.
- **Gerakan Halus (RIFE)**: Fitur opsional interpolasi frame rate 2x menggunakan Practical-RIFE untuk gerakan video yang lebih mulus (misal: 30 fps menjadi 60 fps).
- **Antarmuka Desktop Lokal Gradio dengan Pengganti Bahasa Dinamis**:
  - **Bahasa Inggris Default**: Tampilan modern dan bersih yang dikonfigurasi dalam bahasa Inggris secara default.
  - **Tombol Switch Pojok Kanan Atas**: Tombol switch interaktif di pojok kanan atas header untuk beralih seketika antara **English** dan **Bahasa Indonesia**.
  - **Pembaruan UI Real-Time**: Seluruh label, slider, tooltip info, dropdown, dan tombol langsung terbarui secara reaktif saat bahasa diganti.
  - **Unggah Video Drag-and-Drop**: Progress bar per-frame, preview perbandingan frame tengah sebelum/sesudah, dan tombol unduh MP4 sekali klik.

---

## Antarmuka & Tombol Ganti Bahasa

Aplikasi akan terbuka dalam **Bahasa Inggris** secara default. Kapan pun Anda dapat mengganti bahasa menggunakan switch di pojok kanan atas:

1. Temukan switch **🌐 Language / Bahasa** di pojok kanan atas baris judul aplikasi.
2. Klik opsi **Bahasa Indonesia** untuk mengubah seluruh teks, label, info slider, opsi dropdown, dan pesan status ke Bahasa Indonesia.
3. Klik **English** untuk mengembalikan tampilan ke Bahasa Inggris.
4. Nilai pengaturan yang telah Anda atur tetap terjaga saat beralih bahasa.

---

## Struktur Proyek

```
portrait-video-enhancer/
├── app.py                  # Titik masuk aplikasi, UI Gradio dengan switch bilingual
├── pipeline/
│   ├── __init__.py         # Patch kompatibilitas runtime (torchvision/basicsr)
│   ├── extract.py          # Ekstraksi frame FFmpeg + audio + probe metadata
│   ├── face_restore.py     # Wrapper CodeFormer & GFPGAN dengan offload VRAM
│   ├── upscale.py          # Wrapper Real-ESRGAN tiled fp16
│   ├── interpolate.py      # Interpolasi gerakan RIFE (2x FPS)
│   ├── enhancer.py         # Koordinator pemrosesan frame utama
│   └── reassemble.py       # Penggabung kembali video + audio FFmpeg
├── weights/                # Direktori bobot model
│   ├── CodeFormer/
│   ├── facelib/
│   ├── realesrgan/
│   ├── gfpgan/
│   └── rife/
├── fix_basicsr.py          # Script otomatis perbaikan kompatibilitas BasicSR di PyTorch 2.x
├── download_models.py      # Pengunduh otomatis bobot model dengan resume & progres
├── test_pipeline.py        # Pengujian ekstraksi dan perakitan mandiri
├── portrait_video_enhancer_colab.ipynb # Notebook Google Colab siap pakai (GPU T4 Gratis)
├── requirements.txt        # Daftar dependensi minimal
├── run.bat                 # Script peluncur satu klik (pengaturan env & port)
├── start.bat               # Shortcut peluncur aplikasi
└── README.md               # Dokumentasi bilingual (English & Bahasa Indonesia)
```

---

## Panduan Mulai Cepat & Instalasi

### ⚡ Menjalankan di Google Colab (GPU Cloud Gratis)
Jika komputer Anda tidak memiliki GPU RTX, Anda dapat menjalankan seluruh aplikasi secara online di Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adewanggar/portrait-video-enhancer/blob/main/portrait_video_enhancer_colab.ipynb)

1. Buka notebook melalui tombol di atas.
2. Masuk ke **Runtime** > **Change runtime type** > pilih **T4 GPU** (layanan gratis).
3. Jalankan seluruh cell secara berurutan.
4. Buka tautan publik `*.gradio.live` yang muncul pada cell terakhir.

---

### Instalasi Lokal

### 1. Kebutuhan Sistem
- **Sistem Operasi**: Windows 10/11
- **Python**: Versi 3.10 atau 3.11
- **GPU**: NVIDIA RTX 3050 Laptop GPU (6GB VRAM) atau lebih tinggi
- **FFmpeg**: Terinstal dan terdaftar di `PATH` sistem (teruji pada FFmpeg 9.0+)

### 2. Instalasi Dependensi
```powershell
pip install -r requirements.txt
```

### 3. Jalankan Patch Kompatibilitas BasicSR / PyTorch
BasicSR 1.4.2 memiliki kendala kompatibilitas registry pada PyTorch 2.x. Jalankan perintah perbaikan berikut:
```powershell
python fix_basicsr.py
```

### 4. Unduh Bobot Model
Unduh seluruh bobot model secara otomatis dari repositori resmi:
```powershell
python download_models.py --all
```

Untuk mengunduh kategori tertentu saja:
```powershell
python download_models.py --category codeformer
python download_models.py --category realesrgan
python download_models.py --category gfpgan
python download_models.py --category rife
```

Untuk memeriksa status kelengkapan bobot model lokal:
```powershell
python download_models.py --verify
```

### 5. Menjalankan Aplikasi
Aplikasi dapat dijalankan melalui:
```powershell
run.bat
```
atau langsung melalui Python:
```powershell
python app.py
```
Browser default Anda akan otomatis membuka alamat: `http://127.0.0.1:7860`.

---

## Panduan Optimasi VRAM 6GB

| Fitur | Mode Cepat (Fast) | Mode Kualitas (Quality) |
| :--- | :--- | :--- |
| **Ukuran Tile Real-ESRGAN** | 400 | 300 |
| **Interpolasi Gerak RIFE** | Nonaktif | Opsional (2x FPS) |
| **Presisi Komputasi** | FP16 Half | FP16 Half |
| **Offload VRAM Sekuensial** | Aktif | Aktif |
| **Pembersihan Memori Per-Frame** | `torch.cuda.empty_cache()` | `torch.cuda.empty_cache()` |

### Rincian Alokasi Memori VRAM pada RTX 3050 (6.144 MB)
- **Windows OS + Tampilan Desktop**: ~400 MB
- **Real-ESRGAN (FP16, Tile 400)**: ~2.100 MB
- **CodeFormer + RetinaFace (FP16)**: ~1.800 MB
- **Ruang Aman (Safety Headroom)**: >1.500 MB (bebas dari error OOM)

---

## Penulis & Lisensi

- **Penulis (Author)**: adewanggar
- **Lisensi**: [MIT License](LICENSE)

### Lisensi Model Pihak Ketiga & Penggunaan Komersial

> [!IMPORTANT]
> Aplikasi ini menggunakan beberapa model open-source yang masing-masing memiliki lisensi tersendiri.

- **CodeFormer**: [NTU S-Lab License 1.0](https://github.com/sczhou/CodeFormer/blob/master/LICENSE) — **Hanya untuk Riset Non-Komersial**. Apabila Anda menggunakan aplikasi ini untuk keperluan komersial, ganti pilihan restorasi wajah ke **GFPGAN**.
- **GFPGAN**: [Apache License 2.0](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE) — Lisensi permisif yang mengizinkan penggunaan komersial maupun open-source.
- **Real-ESRGAN**: [BSD 3-Clause License](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) — Bebas digunakan untuk keperluan komersial dan privat dengan mencantumkan atribusi.
- **RIFE (Practical-RIFE)**: [MIT License](https://github.com/hzwer/Practical-RIFE/blob/main/LICENSE) — Lisensi open-source standar yang permisif.
