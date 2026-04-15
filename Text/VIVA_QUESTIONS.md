# EverLearn Vision — Viva Question Bank

Questions are grouped by topic, ordered from basic → tricky.
Each main question has likely follow-ups indented below it.

---

## 1. Project Basics

**Q. What is the purpose of your project? What does it do?**
- What kind of input does it take?
- What does it output?
- Can it classify more than 2 classes?

**Q. What is image classification?**
- How is it different from object detection?
- Where is image classification used in the real world?

**Q. What dataset did you use?**
- How many images?
- How did you split it into train and val?
- Why is splitting important? What happens if you train and test on the same data?

---

## 2. Data Pipeline

**Q. How does your system load the images?**
- What is `ImageFolder`? How does it know which image belongs to which class?
- What happens if two class folders have different names in train and val?

**Q. What is a DataLoader? Why do we need it?**
- Why don't we just load all images at once?
- What is batching? Why batch size 32 specifically?
- What does `shuffle=True` do and why is it only for training?

**Q. What is data augmentation? What augmentations did you apply?**
- Why do we augment training data but NOT validation data?
- What does `RandomHorizontalFlip` actually do to the image?
- What does `ColorJitter` do?

**Q. Why do you normalize images? What values did you use?**
- Where do those specific mean and std values `[0.485, 0.456, 0.406]` come from?
- What happens if you don't normalize?
- After normalization, will pixel values still be between 0 and 1?

**Q. What is `pin_memory` in the DataLoader?**
- Why is it only set to True when CUDA is available?

**Q. What does `is_valid_file` do in `datasets.ImageFolder`?**
- What error would occur if a corrupt image entered the DataLoader?

---

## 3. Model Architecture

**Q. Which model are you using and why?**
- What does ResNet stand for?
- How many layers does ResNet-18 have?
- How many parameters does it have approximately?

**Q. What is Transfer Learning?**
- Why did you use a pretrained model instead of building one from scratch?
- What was the pretrained model originally trained on?
- Which layers did you keep? Which did you change?

**Q. What is the final layer of your model? What did you replace it with?**
- The original ResNet-18 final layer outputs 1000 values. Why?
- You replaced it with a layer that outputs 2 values. What do those 2 values represent?
- What is `nn.Linear`? What does it do mathematically?

**Q. What is a Residual Connection / Skip Connection?**
- What problem does it solve?
- What is the Vanishing Gradient Problem?
- Draw or describe how the skip connection works.

**Q. What are logits?**
- If the model outputs `[3.2, -1.5]` for a 2-class problem, what does that mean?
- How do you convert logits to probabilities?

**Q. What is `model.eval()` vs `model.train()`?**
- What is Dropout? Why is it disabled during evaluation?
- What is BatchNorm? Why does it behave differently in train vs eval mode?

---

## 4. Training Process

**Q. Explain the training loop step by step.**
- What happens in order: zero_grad → forward → loss → backward → step?
- Why do we call `optimizer.zero_grad()` at the start of every batch?
- What happens if you forget to call it?

**Q. What is a loss function? Which one did you use?**
- What is CrossEntropyLoss?
- Does CrossEntropyLoss apply Softmax internally or do you apply it separately?
- What does a loss of 0 mean? What does a high loss mean?

**Q. What is an optimizer? Which one did you use?**
- What is Adam? What does it stand for?
- How is Adam different from plain SGD?
- What is a learning rate?
- What happens if the learning rate is too high? Too low?

**Q. What is backpropagation?**
- What mathematical concept is it based on?
- What does `loss.backward()` do exactly?
- What does `optimizer.step()` do with the gradients?

**Q. What is the Learning Rate Scheduler? Which one did you use?**
- What does `StepLR(step_size=5, gamma=0.5)` do?
- Why do we reduce the learning rate during training instead of keeping it constant?

**Q. What is an epoch?**
- You trained for 10 epochs. Does that mean your model saw each image 10 times?
- How do you know when to stop training?

**Q. How do you save the best model?**
- What is saved inside `model.pth`? Just the weights?
- Why did you also save `class_names` inside the checkpoint?
- What is `model.state_dict()`?

---

## 5. Validation & Metrics

**Q. What is validation accuracy?**
- How is validation different from training?
- Why do we validate after every epoch?

**Q. What does `@torch.no_grad()` do?**
- Why not compute gradients during validation?
- What would happen if you didn't use it?

**Q. How is accuracy calculated in your code?**
- What does `outputs.max(1)` return?
- What does `predicted.eq(labels).sum()` do?

---

## 6. Inference / Web App

**Q. How does the web app work?**
- What framework did you use? Why Streamlit and not Flask?
- What happens after the user uploads an image?

**Q. What is Softmax? Why is it used at inference time?**
- If logits are `[3.2, -1.5]`, what would Softmax output approximately?
- What is the property of Softmax output that logits don't have?

**Q. What is `@st.cache_resource`?**
- Why would not caching the model be a problem?
- How many times is the model loaded with caching vs without?

**Q. What does `predictor.predict()` return?**
- What is the difference between `predict()` and `predict_all_probs()`?

---

## 7. Code Structure & Design

**Q. Why did you separate your code into `src/` and keep `train.py` at the root?**
- What is the benefit of having `dataset.py`, `model.py`, `trainer.py` as separate files?

**Q. What is `config.py` for?**
- Why not just hardcode values like `batch_size=32` directly in `train.py`?

**Q. What does argparse do in `train.py`?**
- Give an example of how you'd change the backbone from the command line.
- What is the default backbone?

**Q. What do the utility scripts do? Are they required?**
- `split_dataset.py` — what does it do and when would you use it?
- `clean_dataset.py` — what error does it prevent?
- `verify_dataset.py` — what is it checking exactly?

---

## 8. Tough / Conceptual Questions

**Q. What is overfitting? Is your model overfitting?**
- How would you detect overfitting from the training logs?
- How does data augmentation help prevent it?

**Q. If you add a completely new class (e.g. rabbits), what do you need to do?**
- Can you just add a `rabbit/` folder and retrain?
- Do you need to modify any code?

**Q. What would happen if your train/ folder has 900 cat images and only 10 dog images?**
- What is class imbalance?
- How would it affect model predictions?
- How can you fix it?

**Q. Your model outputs 2 numbers. How does it decide which class wins?**
- What is argmax?
- Is argmax the same as Softmax?

**Q. What is the difference between a parameter and a hyperparameter?**
- Give 2 examples of each from your project.

**Q. Why did you choose ResNet-18 specifically and not ResNet-50 or EfficientNet?**
- What is the tradeoff between ResNet-18 and ResNet-50?
- When would you choose EfficientNet?

**Q. What does `num_workers=4` do in the DataLoader?**
- What happens if you set it to 0?
- Why might `num_workers > 0` cause issues on Windows?

**Q. What is the shape of one image tensor after it goes through your transform pipeline?**
- Answer: `[3, 224, 224]` — explain what each dimension means.
- What is the shape of one batch? Answer: `[32, 3, 224, 224]`

---

## 9. If Things Go Wrong (Debugging Questions)

**Q. Training accuracy is 99% but validation accuracy is 55%. What does this tell you?**
- What is the fix?

**Q. Your training crashes with `RuntimeError: Expected all tensors to be on the same device`**
- What caused this?
- Where in the code does `.to(device)` need to be called?

**Q. Your model always predicts the same class no matter what image you give it.**
- What could have caused this?
- How would you debug it?

**Q. How would you know if normalization was not applied correctly?**
- What does `dataloader_demo.py` tell you about this?

---

> All answers are in `VIVA_GUIDE.md` and the source files themselves.
> The toughest questions are in Section 8 and 9 — focus on those.
