# 🐔🥚 ChickenOrEgg — Auto Image Classifier & Annotator

> *"Which came first, the label or the annotation?"*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Powered%20by-Ollama-green.svg)](https://ollama.com/)

ChickenOrEgg is a lightweight, **fully local** toolkit for automatically classifying and annotating images using open-source Vision-Language Models (VLMs) and Meta's Segment Anything Model 3 (SAM3). No cloud API required — everything runs on your own machine.

Originally built for newspaper detection, ChickenOrEgg is now a **general-purpose** tool that works for any two-class image sorting task — receipts, product photos, wildlife shots, documents, you name it.

---

## ✨ Features

- **Local VLM Classification** — Uses [Ollama](https://ollama.com/) with any vision model (default: `qwen2.5vl:7b`). Private, offline, free.
- **Any Two Classes** — Just pass `--target-class` and `--other-class`. No code changes needed.
- **All Image Formats** — Supports `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.gif` out of the box.
- **Few-Shot Learning** — Optionally supply example images to boost accuracy via in-context prompting.
- **Batch Processing & Reports** — Automatically sorts images into folders and saves a JSON results file + text report.
- **SAM3 Polygon Annotation** — Generate masks and polygon annotations with SAM3. Colab and local notebook versions included.

---

## 📁 Project Structure

```
ChickenOrEgg/
├── classifier/
│   ├── template_classifier.py   ← Generic classifier (START HERE)
│   ├── image_dis.py             ← Original newspaper-specific classifier
│   └── examples/                ← Sample images for few-shot prompting
├── annotation/
│   ├── sam3_generic_annotator_template_colab.ipynb   ← SAM3 (Colab)
│   └── sam3_generic_annotator_template_local.ipynb   ← SAM3 (Local)
├── requirements.txt
├── .env.example
└── LICENSE
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- (Optional) CUDA GPU for faster SAM3 inference

### 2. Install

```bash
git clone https://github.com/Jinsenloh/ChickenOrEgg.git
cd ChickenOrEgg

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
ollama pull qwen2.5vl:7b
```

### 3. Set up environment (optional)

```bash
cp .env.example .env
# Edit .env and fill in HF_TOKEN if you need to download SAM3
```

---

## 🎯 Usage: Classification

### Generic Classifier (recommended)

```bash
python classifier/template_classifier.py \
    --source-dir ./my_images \
    --output-dir ./output \
    --target-class "Pizza" \
    --other-class "Not Pizza"
```

**All options:**

| Flag | Description | Default |
|---|---|---|
| `--source-dir` | Folder of images to classify | *(required)* |
| `--output-dir` | Where to save results | `./output` |
| `--target-class` | Label for the positive class | `"Target Object"` |
| `--other-class` | Label for the negative class | `"Other Object"` |
| `--model` | Any Ollama vision model tag | `qwen2.5vl:7b` |
| `--file-pattern` | Glob filter e.g. `"*.jpg"` | all image types |
| `--example-class-a` | Positive example image paths (few-shot) | — |
| `--example-class-b` | Negative example image paths (few-shot) | — |
| `-y / --yes` | Skip confirmation prompt | — |

**With few-shot examples:**

```bash
python classifier/template_classifier.py \
    --source-dir ./receipts_raw \
    --output-dir ./receipts_sorted \
    --target-class "Receipt" \
    --other-class "Not Receipt" \
    --example-class-a ./classifier/examples/paper305.jpg ./classifier/examples/paper307.jpg \
    --example-class-b ./classifier/examples/paper3.jpg ./classifier/examples/paper75.jpg
```

**Use a different Ollama model:**

```bash
python classifier/template_classifier.py \
    --source-dir ./images \
    --output-dir ./output \
    --target-class "Cat" \
    --other-class "Not Cat" \
    --model llava:13b
```

**Tip:** Open `template_classifier.py` and edit the `_build_system_prompt()` method to add visual characteristics specific to your classes — this significantly improves accuracy.

### Newspaper Classifier (original)

```bash
python classifier/image_dis.py \
    --source-dir /path/to/images \
    --output-dir /path/to/output
```

---

## 🎭 Usage: Annotation (SAM3)

The `annotation/` notebooks use SAM3 to automatically detect objects, generate segmentation masks, and export polygon data (RLE + bounding boxes) as JSON.

- Use `*_colab.ipynb` versions on Google Colab (includes Drive mounting).
- Use `*_local.ipynb` versions on your own machine.
- Just change the `text_prompt` variable inside any notebook to detect any object type.

**SAM3 requires a Hugging Face token** to download the gated model weights. Add it to `.env`:

```env
HF_TOKEN=your_token_here
```

---

## 📦 Output Format

After classification, your `--output-dir` will contain:

```
output/
├── pizza/                        ← images classified as "Pizza"
├── not_pizza/                    ← images classified as "Not Pizza"
├── classification_results.json   ← full per-image analysis
└── classification_report.txt     ← human-readable summary
```

The JSON file includes per-image confidence scores, model reasoning, and detected visual elements — useful for auditing or downstream processing.

---

## 🔧 Environment Variables

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token — required to download SAM3 model weights |
| `OPENROUTER_API_KEY` | OpenRouter API key — only needed if testing cloud VLM alternatives |

---

## 📄 License

[MIT](LICENSE)
