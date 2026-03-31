# EverLearn Vision

A **dataset-agnostic image classification system** built with Python and PyTorch.  
Drop in any folder-structured dataset and start training in minutes.

---

## Project Structure

```
EverlLearn-Vision/
├── config.py            ← All settings live here
├── train.py             ← Main training entry point
├── verify_dataset.py    ← Validate your dataset before training
├── requirements.txt
├── src/
│   ├── dataset.py       ← DataLoader (auto-detects classes from folders)
│   ├── model.py         ← Pretrained backbones + custom head
│   └── trainer.py       ← Train / evaluate loops
└── data/                ← Your dataset goes here
    ├── train/
    │   ├── class1/
    │   └── class2/
    └── val/
        ├── class1/
        └── class2/
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

## 2 · Prepare Your Dataset (Cats example)

```
data/
├── train/
│   ├── cat/
│   │   ├── cat001.jpg
│   │   └── ...          (≥ 100 images recommended)
│   └── not_cat/
│       └── ...
└── val/
    ├── cat/
    │   └── ...
    └── not_cat/
        └── ...
```

> Add as many class folders as you need — the system detects them automatically.

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

Best model is saved to `checkpoints/best_model.pth`.

---

## 5 · Configuration

Edit [`config.py`](config.py) to change defaults without touching any other file:

| Setting | Default | Description |
|---|---|---|
| `DATA_DIR` | `"data"` | Root dataset folder |
| `IMAGE_SIZE` | `(224, 224)` | Resize resolution |
| `BATCH_SIZE` | `32` | Training batch size |
| `NUM_EPOCHS` | `10` | Training epochs |
| `LEARNING_RATE` | `1e-3` | Initial LR |
