# EverLearn Vision

A **dataset-agnostic, self-improving image classification system** built with PyTorch, FastAPI, and Next.js.

Drop in **any** folder-structured dataset — the system auto-detects classes, trains a model, serves predictions via a REST API, collects user feedback, and retrains itself to improve over time.

> Built and tested on a Cats vs Dogs dataset, but works with any number of classes out of the box.

---

## Features

| Feature | Description |
|---|---|
| 🔄 **Dataset-Agnostic** | Auto-discovers classes from folder names — no code changes to switch datasets |
| 🧠 **Self-Improving** | User feedback is stored and used to retrain the model automatically |
| 🏗️ **Multiple Backbones** | ResNet-18, ResNet-50, EfficientNet-B0, MobileNet-V3 |
| 📊 **MLflow Tracking** | All training runs, metrics, and model artifacts are tracked |
| 🌗 **Modern Frontend** | Next.js app with dark/light mode, animations, and prediction history |
| 🗄️ **Zero-Config Database** | SQLite by default (no setup) — PostgreSQL supported via env var |
| 🔥 **Hot Reload** | Reload the model without restarting the server |

---

## Project Structure

```
EverlLearn-Vision/
├── config.py                  ← All settings (data dir, batch size, LR, etc.)
├── train.py                   ← Main training entry point
├── retrain.py                 ← Self-improving retraining pipeline
├── retrieve_best_model.py     ← Retrieve best model from MLflow
├── requirements.txt
├── app/                       ← FastAPI backend
│   ├── main.py                ← REST API (predict, feedback, retrain, reload)
│   ├── model_loader.py        ← Loads checkpoint into a ModelBundle
│   ├── inference.py           ← Image → prediction pipeline
│   ├── database.py            ← SQLAlchemy engine (SQLite / PostgreSQL)
│   ├── models.py              ← ORM model for feedback table
│   ├── schemas.py             ← Pydantic request/response schemas
│   └── crud.py                ← Database read/write operations
├── src/                       ← ML core
│   ├── dataset.py             ← DataLoader (auto-detects classes from folders)
│   ├── model.py               ← Pretrained backbones + custom head
│   ├── trainer.py             ← Train / evaluate loops
│   └── predictor.py           ← Single-image inference engine
├── frontend/                  ← Next.js web app
│   └── src/app/
│       ├── page.js            ← Main UI
│       └── components/        ← UploadBox, PredictionCard, FeedbackPanel, etc.
├── data/                      ← Your dataset goes here
│   ├── train/
│   │   ├── class_a/
│   │   └── class_b/
│   └── val/
│       ├── class_a/
│       └── class_b/
├── checkpoints/               ← Saved model weights
├── optional dataset files/    ← split_dataset.py, clean_dataset.py, verify_dataset.py
└── optional testing scripts/  ← dataloader_demo.py, model_demo.py
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm (for the frontend)
- A dataset in `data/train/<class>/` and `data/val/<class>/` structure

### 1 · Install Python Dependencies

```bash
cd EverlLearn-Vision

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2 · Prepare Your Dataset

Place your images in the `data/` folder using this structure:

```
data/
├── train/
│   ├── your_class_1/
│   │   ├── image001.jpg
│   │   └── ...
│   └── your_class_2/
│       └── ...
└── val/
    ├── your_class_1/
    │   └── ...
    └── your_class_2/
        └── ...
```

> **Each sub-folder name = one class label.** The system auto-detects them. You can have any number of classes.

**If your data is NOT split yet**, use the included utility:
```bash
python "optional dataset files/split_dataset.py"
```

### 3 · Train the Model

```bash
# Default (ResNet-18, 10 epochs)
python train.py

# Custom options
python train.py --backbone resnet50 --epochs 20 --lr 0.0003 --batch_size 64
```

Available backbones: `resnet18`, `resnet50`, `efficientnet_b0`, `mobilenet_v3_small`

Best model is saved to `checkpoints/model.pth` with class names baked in.

### 4 · Start the Backend (FastAPI)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check: [http://localhost:8000/](http://localhost:8000/)

### 5 · Start the Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) — upload any image to classify it, view confidence scores, and submit feedback.

### 6 · View MLflow Dashboard (Optional)

```bash
mlflow ui --port 5000
```

Open [http://localhost:5000](http://localhost:5000) to view training runs, compare metrics, and download model artifacts.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + model metadata (classes, backbone, version) |
| `POST` | `/predict` | Upload an image → get predicted label + confidence |
| `POST` | `/feedback` | Submit a user correction for a prediction |
| `GET` | `/feedback` | Retrieve all stored feedback (paginated) |
| `POST` | `/reload-model` | Hot-reload the latest model without restarting |
| `POST` | `/retrain` | Trigger retraining pipeline in the background |

---

## Self-Improving Pipeline

```
User uploads image → Model predicts → User corrects (feedback)
                                            ↓
            API stores correction in database (SQLite/PostgreSQL)
                                            ↓
                POST /retrain → Augment dataset with corrections
                                            ↓
               Fine-tune model → Compare accuracy → Promote if better
                                            ↓
                    POST /reload-model → Serve improved model
```

### Trigger Retraining

```bash
# Via API
curl -X POST http://localhost:8000/retrain

# Or directly
python retrain.py --epochs 5 --lr 0.0001
```

---

## Switching Datasets

### Replace the dataset and retrain

```bash
# 1. Remove old data
rm -rf data/train data/val

# 2. Add your new dataset (same folder structure as above)
#    Each subfolder = one class

# 3. Delete the old checkpoint (important — it was trained on different classes)
rm -f checkpoints/model.pth

# 4. Train on the new dataset
python train.py
# The system auto-detects your new class names from folder names
```

> **How class detection works:** The system scans sub-folder names inside `data/train/` at runtime. Each folder = one class. Class names are baked into the checkpoint at save time, so inference always knows what it's predicting — no config needed.

---

## Database Configuration

By default, EverLearn Vision uses **SQLite** — a zero-config file-based database (`everlearn.db`). No setup required.

To use **PostgreSQL** instead:

```bash
export DATABASE_URL=postgresql://user:password@localhost:5432/everlearn_vision
pip install psycopg2-binary
```

---

## Configuration

Edit [`config.py`](config.py) to change defaults without touching any other file:

| Setting | Default | Description |
|---|---|---|
| `DATA_DIR` | `"data"` | Root dataset folder |
| `IMAGE_SIZE` | `(224, 224)` | Resize resolution |
| `BATCH_SIZE` | `32` | Training batch size |
| `NUM_EPOCHS` | `10` | Training epochs |
| `LEARNING_RATE` | `1e-3` | Initial LR |
| `DEVICE` | `"auto"` | Auto-detected (CUDA → MPS → CPU) |

---

## All Commands at a Glance

```bash
# ── Setup ──────────────────────────────────────
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# ── Train ──────────────────────────────────────
python train.py
python train.py --backbone resnet50 --epochs 20

# ── Backend ────────────────────────────────────
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# ── Frontend ───────────────────────────────────
cd frontend && npm install && npm run dev

# ── MLflow ─────────────────────────────────────
mlflow ui --port 5000

# ── Retrain (self-improving) ───────────────────
python retrain.py
curl -X POST http://localhost:8000/retrain

# ── Hot-reload model ───────────────────────────
curl -X POST http://localhost:8000/reload-model

# ── Dataset utilities (optional) ───────────────
python "optional dataset files/verify_dataset.py"
python "optional dataset files/clean_dataset.py"
python "optional dataset files/split_dataset.py"
```
