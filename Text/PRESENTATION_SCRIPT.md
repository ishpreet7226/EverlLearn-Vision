# EverLearn Vision — Viva Introduction Script
### (~5–7 minutes, conversational tone, student speaking to teacher)

---

Sir, my project is called **EverLearn Vision**.
It is an **image classification system** — meaning you give it an image,
and it tells you which category that image belongs to.

For example, I trained it on a cats and dogs dataset —
so if you upload a picture of a cat, it tells you "cat" with a confidence percentage.
But the system is built in a way that you can swap out the dataset for anything —
plant diseases, handwritten digits, any set of images organised in folders —
and it will work without changing any code.

---

### What problem does it solve?

The idea is simple — classifying images manually is not scalable.
If you have thousands of images, a human can't label them one by one.
So this system uses a trained deep learning model to automate that.

---

### How does it work — end to end?

The workflow has three main stages: **data preparation, training, and inference.**

**Stage 1 — Data Preparation**

Your images go into a folder structure like this:
```
data/
  train/
    cats/   → all cat images
    dogs/   → all dog images
  val/
    cats/
    dogs/
```
Each folder name becomes the class label automatically.
If the data isn't already split, I have a script called `split_dataset.py`
that randomly moves 20% of images into the `val/` folder for testing.
I also have `clean_dataset.py` which scans for corrupt images before training starts,
because even one broken image will crash the DataLoader mid-training.
And `verify_dataset.py` just prints a table confirming the counts and folder structure — like a sanity check.

These are all optional helper scripts — the main pipeline is `train.py`.

---

**Stage 2 — Training**

When I run `python train.py`, it does these things in sequence:

First, it loads the images using PyTorch's `DataLoader`.
Images are loaded in batches — 32 at a time — which is more efficient than processing one image at a time.
Before the images go into the model, they pass through a **transform pipeline**:
they get resized to 224×224 pixels, randomly flipped horizontally, and color-jittered slightly.
This is called **data augmentation** — it prevents the model from memorising exact images
and forces it to learn general features instead.
The images are also normalised using specific mean and standard deviation values from ImageNet,
because the model we're using was originally trained on ImageNet.

Now for the model itself — I'm using **ResNet-18**.
ResNet stands for Residual Network, created by Microsoft Research.

I'm not training it from scratch. I'm using **Transfer Learning** —
which means I take a model that was already trained on 1.2 million images from the ImageNet dataset,
and I reuse all of that learning.
The model already knows how to detect basic features like edges, shapes, and textures.
I only replace the final classification layer — changing it from predicting 1000 ImageNet categories
to predicting my 2 categories — cats and dogs.
This way I need far less data and far less training time.

ResNet has a specific architectural feature called **Residual Connections** or Skip Connections.
In normal neural networks, as you add more layers, gradients tend to become very small
during backpropagation and the model stops learning — this is the Vanishing Gradient Problem.
ResNet solves this by adding the input of a block directly to the output of that block,
creating a shortcut path for gradients to flow through.
That's what allows it to have 18 layers and still train effectively.

For training, I use:
- **CrossEntropyLoss** as the loss function — it measures how wrong the model's prediction is for each batch.
- **Adam optimizer** — it uses the error signal to adjust the model weights after every batch.
  I use Adam over basic SGD because Adam adapts the learning rate per parameter automatically,
  which makes it converge faster.
- **StepLR scheduler** — it halves the learning rate every 5 epochs.
  Early in training the model needs big updates; later it needs small precise ones.
  The scheduler handles that automatically.

After every epoch, if the validation accuracy improves, the model is saved to `checkpoints/model.pth`.
The saved file includes not just the weights but also the class names —
so the app knows what it's predicting without needing any separate config file.

---

**Stage 3 — Inference (the Web App)**

Once training is done, I run `streamlit run app.py`.
Streamlit is a Python library that lets you build a web interface in pure Python.
No HTML, no JavaScript needed — it runs in the browser on port 8501.

On the web page, you upload any image.
The app passes it through the same 224×224 resize and normalise pipeline,
runs it through the saved ResNet model,
and applies **Softmax** to convert raw prediction scores into probabilities that add up to 100%.
The predicted class and confidence percentage are displayed, along with bars for every class.

The model is only loaded once using Streamlit's `@st.cache_resource` decorator —
otherwise Streamlit reruns the script on every user interaction and would reload it every time.

---

### Libraries used and why

| Library | What I use it for | Why this and not alternatives |
|---|---|---|
| **PyTorch** | Core deep learning — tensors, model, training loop | Most widely used in academic research, easy to debug, dynamic computation graph |
| **Torchvision** | Pre-trained ResNet model + ImageFolder + transforms | Tightly integrated with PyTorch, handles folder-based datasets natively |
| **Pillow (PIL)** | Open and validate image files | Standard Python image library, used for corrupt-file detection |
| **Streamlit** | Web interface for predictions | Builds a full web app in pure Python — no frontend code needed |
| **tqdm** | Progress bar during training | Shows batch-level progress so I can monitor training in real time |
| **argparse** | Command-line arguments for train.py | Built into Python, lets me change epochs/backbone/lr without editing the file |

---

### Project files in brief

- `config.py` — all default settings in one place (learning rate, batch size, image size, etc.)
- `src/dataset.py` — loads and transforms images, builds the DataLoader
- `src/model.py` — builds ResNet and replaces the final layer for our classes
- `src/trainer.py` — contains the train and validate loops
- `src/predictor.py` — loads the saved model and runs prediction on a single image
- `train.py` — the main script that ties everything together
- `app.py` — the Streamlit web interface

---

That's essentially the full project, sir.
Happy to go into any part in more detail.

---
> **Reading pace note:** This is approximately 750–800 words. At a normal conversational pace
> (not rushing, not slow) that's about 5–6 minutes.
