# TiamaT – Toolkit for Integrated Annotation and Machine-learning Assisted Training

![Python](https://img.shields.io/badge/Python-3.10–3.13-blue)
![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-green)

**TiamaT** is a complete, modular pipeline that transforms raw, unstructured images into fully annotated, machine learning–ready datasets.

Rather than evaluating a model on a fixed test corpus, TiamaT integrates manual correction of predictions into a continuous cycle of training and evaluation. This approach keeps human expertise at the heart of computational processing — which is especially suited to heritage corpora, for which there is no ideal ground truth defined in advance.

Originally built for historical document analysis, TiamaT fits any project where annotations are built incrementally or interactively.

> The name is a nod to [**Tiamat**](https://en.wikipedia.org/wiki/Tiamat), the Mesopotamian goddess of the ocean and chaos — an appropriate symbol for turning raw data into structured knowledge.

**→ Full documentation and installation guide: [tiamat-app.github.io](https://tiamat-app.github.io)**

## ⚠️ Design note

TiamaT is fully functional and ready to use. A design overhaul (CSS, layout, responsive) 
is currently in progress. If you encounter display issues, they will be addressed in an upcoming release.

---

## Table of Contents

- [Workflow Overview](#workflow-overview)
- [Installation](#installation)
- [License & Attribution](#license--attribution)

---

## Workflow Overview

TiamaT follows a structured, iterative cycle. Each pass through the pipeline refines the model using your corrections as new ground truth. The process repeats until results meet your research criteria. You can also download annotated images, structured datasets, and performance metrics for publication or further analysis.

| Step | Description | Tool |
|------|-------------|------|
| **01 Import your corpus** | Load your image collection — manuscripts, photographs, museum records — into TiamaT via the interface. | |
| **02 Annotate a sample** | Draw bounding boxes around the visual elements you want to detect: miniatures, iconographic motifs, decorative elements… | Label Studio |
| **03 Evaluate your dataset** | Before training, TiamaT provides a quality overview of your annotations: class distribution, ratio of annotated vs. unannotated images, and potential imbalances to address. | |
| **04 Train the model** | Configure key parameters — epochs, batch size, image size — and optionally fine-tune advanced settings. Start from scratch or build on an existing model, including one previously trained within TiamaT. | YOLO |
| **05 Run inference** | Apply the trained model to new, unannotated images. TiamaT generates predictions automatically for each one. | YOLO |
| **06 Review & correct** | Examine predictions in the annotation interface, correct errors, and validate results. | Label Studio |
| **07 Evaluate the model** | Based on predictions and corrected annotations, TiamaT generates performance metrics — precision, recall, F1 score, and confusion matrix. | |
| **08 Iterate** | Corrected annotations are added to the training set. Repeat the cycle until the model's performance meets your research criteria. | |

---

## Installation

> **Python 3.10–3.13 required.** Python 3.14 is not yet supported due to a Label Studio compatibility issue.

### 1. Prerequisites

Check your Python version:

```bash
python3 --version
```

macOS users: if Python is not installed, we recommend using [Homebrew](https://brew.sh) — `brew install python3`.

### 2. Clone the repository

**HTTPS:**
```bash
git clone https://github.com/TiamaT-app/TiamaT_app.git
cd TiamaT_app
```

**SSH:**
```bash
git clone git@github.com:TiamaT-app/TiamaT_app.git
cd TiamaT_app
```

### 3. Install dependencies

**macOS / Linux:**
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python3 -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

#### PyTorch installation

PyTorch must be installed separately depending on your hardware.
Visit https://pytorch.org/get-started/locally/ and select your configuration
to get the appropriate installation command.

Versions tested with TiamaT: `torch==2.12.1`, `torchvision==0.27.1`


### 4. Launch the application

**macOS / Linux:**
```bash
source env/bin/activate
python3 run.py
```

**Windows:**
```bash
.venv\Scripts\activate
python3 run.py
```

TiamaT will open in your browser at `http://127.0.0.1:5001`

---

## License & Attribution

Any use, even partial, of the content in this repository must be accompanied by proper citation.

**Made with ❤️ by [Marion Charpier](https://github.com/Chaouabti/) & [Fantin Le Ber](https://github.com/fant1-LB/)**  
© 2023–2026 • Project **TiamaT** — Developed within the TORNE-H project, ENC-PSL  
Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)