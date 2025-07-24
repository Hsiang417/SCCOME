# SCCOME: Scene Change Capture and Optical Motion Estimation for Video Quality Assessment

This repository contains the official implementation of **SCCOME**, a no-reference video quality assessment (NR-VQA) framework that incorporates scene change detection, optical flow-based motion analysis, and multi-modal feature fusion.

GitHub Repo: [https://github.com/Hsiang417/SCCOME/tree/main](https://github.com/Hsiang417/SCCOME/tree/main)

---

## 📦 Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure you have PyTorch (with GPU support) installed.

---

## 🚀 Usage Workflow

The pipeline is divided into three main stages:

### 1. Dataset Info Extraction

Use scripts in the `data/` folder to extract metadata (e.g., frame counts, labels, and max sequence length) for each dataset:

```bash
python data/Kon.m                 # for KoNViD-1k
python data/CVD2014.m             # for CVD2014
python data/LIVE_Qualcomm.m       # for LIVE-Qualcomm
```

This will generate `.mat` files like `KoNViD-1kinfo.mat` containing index lists and normalization parameters.

---

### 2. Feature Extraction

Extract spatial-temporal features using:

```bash
python SCCOME_extract.py --database KoNViD-1k
```

Extracted features will be saved as `.npy` files in `CNN_features_{dataset}` folders, with filenames like:

```
{id}_RGBcannyOptreplacedconvnext_3Dmaxmeanstd_features.npy
```

These represent RGB-Canny merged features combined with optical flow and ConvNeXt + 3D statistics.

---

### 3. Quality Score Prediction

Run training and evaluation:

```bash
python SCCOME.py --database KoNViD-1k --exp_id 0
```

Key arguments:

- `--cross Y` for cross-dataset testing
- `--model score_test` specifies model variant
- `--epochs`, `--batch_size`, `--lr` for training control

Checkpoints and results will be saved to:

- `models/`
- `results/`

---

## 📁 Folder Overview

```
├── data/                       # Scripts to prepare dataset info
├── CNN_features_KoNViD-1k/     # Saved feature files (output)
├── SCCOME_extract.py           # Feature extraction
├── SCCOME.py                   # Training & evaluation
├── models/                     # Saved model weights
├── results/                    # Evaluation results
```

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@misc{liao2025sccome,
  title = {SCCOME: Scene Change Capture and Optical Motion Estimation for Video Quality Assessment},
  author = {Haoshiang Liao},
  year = {2025},
  note = {\url{https://github.com/Hsiang417/SCCOME}}
}
```

---

## 📬 Contact

For questions or feedback, please open an [issue](https://github.com/Hsiang417/SCCOME/issues) or reach out via GitHub.

---

