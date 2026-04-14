# EverLearn Vision - Comprehensive Viva Defense Guide

This document is your ultimate guide to understanding every single line of code, technology, and architectural decision used in the **EverLearn Vision** project. It is structured to help you answer any cross-question during your viva defense.

---

## 1. Core Technologies & Stack Selection

During a viva, examiners will ask: *"Why did you choose this technology stack over alternatives like TensorFlow/Keras or a pure drag-and-drop builder?"*

| Technology | Why We Used It | Why Not Alternatives? |
| --- | --- | --- |
| **PyTorch (`torch`)** | PyTorch is the industry and academic standard for deep learning. It uses a "dynamic computation graph" (eager execution), meaning the graph is built on the fly as operations occur. This makes debugging incredibly easy because you can print individual tensors directly at any step. | TensorFlow/Keras uses static graphs which are harder to debug. PyTorch gives us granular control over the training loop, loss computation, and optimizer updates. |
| **Torchvision** | Specifically built to integrate with PyTorch, it provides pre-trained models (ResNet, EfficientNet) and dataset utilities (`ImageFolder`, `transforms`). We use it to avoid writing boilerplate code for dataset loading and image augmentation. | Writing a custom image loading script using pure `cv2` or `PIL` would be slow and prone to errors. `ImageFolder` automatically handles folder-to-class labeling. |
| **Pillow (`PIL`)** | The standard Python Imaging Library. We use it to read raw image files from the disk and ensure they aren't corrupt. | OpenCV (`cv2`) is heavier, harder to install, and uses BGR format by default. PIL uses standard RGB format which Torchvision expects. |
| **Streamlit** | We use this to build our interactive web interface in pure Python. It allows users to upload images and see predictions instantly without writing any HTML/CSS/JS. | Flask or Django would require setting up separate REST APIs, writing HTML/JS frontend templates, and managing HTTP requests. Streamlit does all of this in a single file recursively. |
| **scikit-learn / matplotlib** | Listed in `requirements.txt` for future extensions (e.g., plotting confusion matrices or doing stratified splits). Our current `split_dataset.py` intentionally uses Python's built-in `random` and `shutil` for zero-dependency simplicity. | Using built-in `random` is faster for simple random shuffling, whereas scikit-learn would be used if we needed highly advanced stratified sampling. |

---

## 2. Step-by-Step System Flow (How data moves)

If asked *"How does the system work from input to output?"*, explain this flow:
1. **Raw Folder** -> `split_dataset.py` randomly splits images into 80% Train and 20% Validation folders.
2. **DataLoader** -> `src/dataset.py` reads these images, resizes them to 224x224, applies random augmentations, normalizes their colors, and groups them into batches.
3. **Model Initialization** -> `src/model.py` downloads a pre-trained "ResNet18", chops off its final layer, and slaps on a fresh, customized, randomized linear layer for our specific number of classes.
4. **Training Loop** -> `src/trainer.py` feeds the batches into the model, calculates how wrong the predictions are (Loss), and slightly adjusts the model's weights using backpropagation (Adam Optimizer) to fix the errors.
5. **Inference/Web App** -> `app.py` loads the best saved `.pth` checkpoint, takes a user-uploaded image, runs a "forward pass", applies Softmax, and outputs human-readable probabilities.

---

## 3. File-by-File Detailed Explanation

Here we break down what happens in each file, step-by-step.

### A) `config.py`
**What it does:** Centralized file holding all global constants (hyperparameters). 
**Why it's done:** Instead of hardcoding numbers inside multiple different files, we put them all here. If you need to change learning rate, batch size or image dimensions, you change it once here and it propagates everywhere.
**Viva Point:** *"We separated hyperparameters from application logic to follow the DRY (Don't Repeat Yourself) principle."*

### B) `src/dataset.py` (The Data Pipeline)
*The examiner asks: "How are you dealing with training data?"*

1. **`is_valid_image(path)`:** Uses PIL to test if an image is readable. **Why:** If the model encounters a broken file mid-training, it will crash. This acts as a shield.
2. **`get_transforms(split)`:** This is our data augmentation pipeline.
    - **Training Transforms:** We apply `RandomResizedCrop`, `RandomHorizontalFlip`, and `ColorJitter`. **Why:** This artificially generates new data. If the model sees a dog looking left, we flip it horizontally so the model learns that a dog looking right is still a dog. This prevents **overfitting** (the model memorizing exact photos instead of learning concepts).
    - **Validation Transforms:** We ONLY apply `Resize` and `CenterCrop`. **Why:** During validation, we need an objective, repeatable test score. We don't want randomness here.
    - **Normalization:** `Normalize(mean=[0.485...], std=[0.229...])`. **Why:** These are the exact mathematical averages of the `ImageNet` dataset (the 1.2 million images our model was originally trained on). Doing this ensures our new images mathematically 'look' identical to the data the ResNet backbone is used to seeing.
3. **`get_dataloaders()`:** Wraps `transforms` into `torchvision.datasets.ImageFolder` and builds a `DataLoader`. 
    - **Why ImageFolder?** It automatically maps folder names to integer labels (e.g. `cats/` -> 0, `dogs/` -> 1).
    - **Why DataLoader/Batching?** We process 32 images at a time (`BATCH_SIZE`) instead of 1 by 1 or all at once. Processing 1 is too slow; processing all at once exceeds GPU memory limits.

### C) `src/model.py` (The Brain)
*The examiner asks: "What model architecture are you using and why not build one from scratch?"*

1. **Transfer Learning (`models.resnet18(weights='DEFAULT')`)**: Our project imports ResNet18 (Residual Networks).
    - **Why not train from scratch?** Because training a high-quality vision model from pure scratch requires millions of images and weeks of GPU compute.
    - **How Transfer Learning Works:** A pre-trained ResNet already understands basic universal concepts like "edges, shapes, colors, lines, and textures." We KEEP all those layers (the **backbone**). We only chop off the VERY LAST layer and replace it with a new layer `nn.Linear(in_features, num_classes)`. Our project is only training this final layer while fine-tuning the rest.
2. **Residual Connections (ResNet):**
    - Normally, data flows straight through a neural network. As the network gets deeper, updates (gradients) get smaller and smaller until they vanish ("Vanishing Gradient Problem").
    - **ResNet's Solution:** It uses a "Skip Connection". The original input jumps *over* layers and is added directly back to the output of the block further down. This allows the network to be incredibly deep (18, 50, 100+ layers) without losing signal strength.

### D) `src/trainer.py` (Learning Logic)
*The examiner asks: "Explain the mathematics of how your model learns."*

1. **`train_one_epoch()`:** This iterates over batches in the DataLoader. It executes the core supervised learning process:
    - **`optimizer.zero_grad()`:** Deletes the old update gradients from the previous batch so they don't pile up.
    - **`outputs = model(images)` (Forward Pass):** The model guesses what the images are. The output is raw numbers called **logits** (e.g. `[12.5, -4.2]`).
    - **`loss = criterion(outputs, labels)`:** We use **CrossEntropyLoss**. This compares the model's guess against the true label. The output is a single number representing "How terrible the guess was". 
    - **`loss.backward()` (Backpropagation):** The system uses Calculus (the Chain Rule) traveling backwards through the model layers. It calculates the exact gradient (direction) every single weight needs to shift to ensure the `loss` drops closer to zero next time.
    - **`optimizer.step()`:** Takes a tiny step applying those calculations.
2. **`@torch.no_grad()` inside `evaluate()`:** Turns off gradient tracking. **Why:** Finding gradients takes massive amounts of memory. During validation, the model is only *testing*, not *learning*, so turning this off makes evaluation much faster.

### E) `train.py` (The Orchestrator)
This is the master script unifying dataset, model, and trainer.
1. **Argparse:** Allows us to dynamically change variables from the terminal (e.g. `python train.py --epochs 20`) without rewriting code.
2. **Loss Function (`nn.CrossEntropyLoss`):** Our chosen metric for multi-class mismatch.
3. **Optimizer (`torch.optim.Adam`):** 
    - *Why Adam and not standard SGD (Stochastic Gradient Descent)?* Adam keeps a running average of successful gradients (Momentum) and dynamically adapts the learning rate for every single parameter. It is vastly faster to converge than SGD.
4. **Learning Rate Scheduler (`StepLR`):** Halves the learning rate every 5 epochs. **Why?** When getting close to the optimal solution, big learning steps cause the model to jump straight over the optimal minimum. Small learning steps guide it precisely into the minimum.
5. **Checkpointing (`torch.save`):** Saves our finished weights and class names into `checkpoints/model.pth` so they can be loaded instantly in the future.

### F) `app.py` (Web Frontend Inference)
1. Loads the `.pth` checkpoint using the custom `Predictor` class.
2. **`@st.cache_resource`:** Streamlit reruns the script from top to bottom every time a user does anything (clicks a button). If we didn't use `cache_resource`, it would try to load the heavy model back into memory every single click. Caching ensures the model is loaded just once.
3. **Softmax Conversion:** The model outputs raw logits. Softmax guarantees the sum of all predictions equals 1.0 (or 100%).

### G) Utility & Demo Scripts
The project includes several utilities to verify its behavior before the heavy training process begins:
- **`verify_dataset.py` & `clean_dataset.py`:** Ensures the datasets are structurally valid and deletes corrupt images using PIL. If corrupt images reach the DataLoader, training crashes instantly.
- **`split_dataset.py`:** Automatically carves out a validation set from your training folders using standard Python `random` and `shutil.move()`.
- **`dataloader_demo.py` & `model_demo.py`:** Quick diagnostic scripts. They let you instantly verify that images are normalized properly and that the ResNet architecture outputs the correct tensor shapes, completely isolating bugs.


---

## 4. Quick-Fire Viva Questions

**Q: What happens if your training data is completely unbalanced (e.g. 10,000 cats and 10 dogs)?**
A: The model will suffer from bias. It will just blindly guess "cat" every time to optimize loss. We solve this by oversampling dogs or using weighted CrossEntropyLoss.

**Q: What is a tensor?**
A: A multi-dimensional matrix containing uniform primitives (like floats). Unlike standard python lists, Tensors map directly to GPU arithmetic logic units for extreme parallel processing.

**Q: Why do we normalize images?**
A: So parameter weights all exist on the same scale, making backpropagation stable and stopping exploding gradients.

**Q: What is Batch Size and why is it 32?**
A: Batch size is the number of images processed before updating weights. A batch size of 1 takes too long to bounce toward the minimum. A batch size of the entire dataset overloads our GPU RAM. 32 guarantees smooth, robust gradient steps while fitting comfortably in memory.
