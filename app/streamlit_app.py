import streamlit as st
import torch
import sys
import os
import tempfile

# Allow src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.cnn_transformer import CNNTransformer
from src.preprocessing.spectrogram import create_mel_spectrogram

st.set_page_config(page_title="Heart Murmur Detection", layout="centered")

st.title("🫀 Heart Murmur Detection")
st.write("Upload a heart sound (.wav) file to detect Normal or Abnormal.")

# Load model once
@st.cache_resource
def load_model():
    model = CNNTransformer()

    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "saved_models", "cnn_transformer_best.pth")
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Analyzing..."):
        features = create_mel_spectrogram(tmp_path)

        features = torch.tensor(features).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            output = model(features)
            prob = torch.sigmoid(output)
            confidence = prob.item()

        if confidence >= 0.5:
            st.error("Prediction: Abnormal")
        else:
            st.success("Prediction: Normal")

        st.write(f"Confidence: {confidence:.4f}")