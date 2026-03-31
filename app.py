"""
EverLearn Vision – Streamlit Web App
=====================================
Upload any image and get an instant prediction with confidence scores.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path
import streamlit as st
from PIL import Image

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.predictor import Predictor

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EverLearn Vision",
    page_icon="🔍",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 EverLearn Vision")
st.markdown("**Dataset-agnostic image classifier** — upload any image to classify it.")
st.divider()

# ── Checkpoint path ───────────────────────────────────────────────────────────
CHECKPOINT = "checkpoints/model.pth"

# Cache the model so it loads only once across re-runs
@st.cache_resource(show_spinner="Loading model...")
def load_predictor():
    return Predictor(CHECKPOINT)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ About")
    st.markdown(
        "**EverLearn Vision** uses a pretrained ResNet-18 fine-tuned on your dataset.\n\n"
        "**Prediction pipeline:**\n"
        "1. Upload image\n"
        "2. Resize & normalise to 224×224\n"
        "3. Forward pass through ResNet\n"
        "4. Softmax → probabilities\n"
        "5. Display top prediction"
    )
    st.divider()
    st.markdown(f"**Checkpoint:** `{CHECKPOINT}`")

# ── Main UI ───────────────────────────────────────────────────────────────────
# Check checkpoint exists before showing uploader
if not Path(CHECKPOINT).exists():
    st.error(
        f"⚠️ No checkpoint found at `{CHECKPOINT}`.\n\n"
        "Please run training first:\n```\npython train.py\n```"
    )
    st.stop()

# Load model
predictor = load_predictor()

st.markdown(
    f"**Classes:** {' · '.join(f'`{c}`' for c in predictor.class_names)}"
)
st.divider()

# File uploader
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, PNG, BMP, WEBP",
)

if uploaded:
    # ── Two-column layout: image | results ────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📷 Preview")
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
        st.caption(f"Size: {image.width} × {image.height} px")

    with col2:
        st.subheader("🏷️ Prediction")

        with st.spinner("Classifying..."):
            # Save upload to a temp path so Predictor can open it
            tmp_path = "/tmp/everlearn_upload.jpg"
            image.save(tmp_path)

            label, confidence = predictor.predict(tmp_path)
            all_probs = predictor.predict_all_probs(tmp_path)

        # ── Result display ────────────────────────────────────────────────────
        color = "green" if confidence >= 0.75 else "orange"
        st.markdown(
            f"### :{color}[{label.upper()}]"
        )
        st.metric(
            label="Confidence",
            value=f"{confidence * 100:.1f}%",
        )

        st.divider()

        # ── Probability bars for all classes ──────────────────────────────────
        st.markdown("**All class probabilities:**")
        for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            pct = prob * 100
            # Highlight the winning class
            label_str = f"**{cls}** ← predicted" if cls == label else cls
            st.write(label_str)
            st.progress(prob, text=f"{pct:.1f}%")

    st.divider()
    st.success("✅ Prediction complete!")
else:
    # Placeholder when no image uploaded yet
    st.info("👆 Upload an image above to get started.")
    st.markdown(
        "**How it works:**\n"
        "1. Your image is resized to 224×224 and normalised\n"
        "2. It passes through all ResNet-18 layers (forward pass)\n"
        "3. The model outputs a score per class (logits)\n"
        "4. Softmax converts scores → probabilities\n"
        "5. The class with the highest probability wins!"
    )
