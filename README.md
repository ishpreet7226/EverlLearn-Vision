# EverLearn Vision

A **dataset-agnostic image classification system** built with Python and PyTorch.  
Drop in **any** folder-structured dataset and start training in minutes — no code changes needed.

> Built and tested on a Cats vs Dogs dataset, but works with any number of classes out of the box.

---

## Project Structure

```
EverlLearn-Vision/
├── config.py              ← All settings live here
├── train.py               ← Main training entry point
├── app.py                 ← Streamlit web app for predictions
├── verify_dataset.py      ← Validate your dataset before training
├── clean_dataset.py       ← Remove corrupt/unreadable images
├── split_dataset.py       ← Auto split raw images into train/val
├── requirements.txt
├── src/
│   ├── dataset.py         ← DataLoader (auto-detects classes from folders)
│   ├── model.py           ← Pretrained backbones + custom head
│   ├── trainer.py         ← Train / evaluate loops
│   └── predictor.py       ← Single-image inference engine
└── data/                  ← Your dataset goes here
    ├── train/
    │   ├── class_a/
    │   └── class_b/
    └── val/
        ├── class_a/
        └── class_b/
```

---

## 1 · Setup

```bash
# Clone or open the project folder
cd EverlLearn-Vision

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2 · Switch or Replace Your Dataset

> Whether you want to **replace the existing cats/dogs dataset** or start fresh with something new — follow these steps.

### Option A — Keep the existing dataset, just retrain

Skip to [Step 3](#3--verify-dataset). No changes needed.

---

### Option B — Delete existing dataset and use a new one

#### Step 1: Delete the current dataset
```bash
# Remove everything inside data/ (keeps the folder)
rm -rf data/train data/val
```

#### Step 2: Add your new dataset

**If your data is already split into train/val:**
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
Just copy your folders into `data/train/` and `data/val/`.

**If your data is NOT split yet (all images in one folder):**
```
data/
└── train/
    ├── your_class_1/
    │   └── ...
    └── your_class_2/
        └── ...
```
Then run the auto-splitter (80% train / 20% val by default):
```bash
python split_dataset.py
# Custom split ratio:
python split_dataset.py --val_split 0.15
```

#### Step 3: Delete the old checkpoint (important!)
The old checkpoint is trained on a different set of classes and **must** be removed before retraining:
```bash
rm -f checkpoints/model.pth
```

#### Step 4: Clean corrupt images (optional but recommended)
```bash
# Dry run — shows bad files without deleting
python clean_dataset.py

# Actually delete corrupt files
python clean_dataset.py --delete
```

#### Step 5: Verify the new dataset structure
```bash
python verify_dataset.py
```

#### Step 6: Train on the new dataset
```bash
python train.py
# The system auto-detects your new class names from folder names
```

---

## 3 · Verify Dataset

```bash
python verify_dataset.py
# Custom path:
python verify_dataset.py --data_dir /path/to/your/data
```

---

## 4 · Train

```bash
# Default (ResNet-18, 10 epochs)
python train.py

# Custom options
python train.py --backbone resnet50 --epochs 20 --lr 0.0003 --batch_size 64
```

Available backbones: `resnet18`, `resnet50`, `efficientnet_b0`, `mobilenet_v3_small`

Best model is saved to `checkpoints/model.pth`.

---

## 5 · Run the Web App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) — upload any image to get an instant prediction with confidence scores for all classes.

---

## 6 · Configuration

Edit [`config.py`](config.py) to change defaults without touching any other file:

| Setting | Default | Description |
|---|---|---|
| `DATA_DIR` | `"data"` | Root dataset folder |
| `IMAGE_SIZE` | `(224, 224)` | Resize resolution |
| `BATCH_SIZE` | `32` | Training batch size |
| `NUM_EPOCHS` | `10` | Training epochs |
| `LEARNING_RATE` | `1e-3` | Initial LR |

---

## 7 · How Class Detection Works

- The system scans sub-folder names inside `data/train/` at runtime
- **Each folder = one class** — name them whatever you like
- Class names are **baked into the checkpoint** at save time, so the web app always knows what it's predicting — no config needed

> Example: folders named `healthy/` and `diseased/` → model predicts `"healthy"` or `"diseased"` automatically.
