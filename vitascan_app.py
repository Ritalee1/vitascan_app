import streamlit as st
import numpy as np
import librosa
import joblib
import os
import soundfile as sf
from pathlib import Path

# === Page configuration ===
st.set_page_config(
    page_title="VitaScan™",
    page_icon="🧠",
    layout="centered"
)

# === Branding with Centered Logo and Titles ===
logo_path = "logo.png"
if Path(logo_path).is_file():
    st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{Path(logo_path).read_bytes().hex()}" width="100">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>VitaScan™</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI-Powered Respiratory Screening</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>🗣️ Listen to your health. Detect early. Act wisely.</p>", unsafe_allow_html=True)

# === Vision Statement ===
st.markdown("""
> 🩺 **Vision**: To empower early detection of respiratory illnesses through AI-powered voice screening — making diagnostics more accessible, non-invasive, and scalable for communities worldwide.
""")

# === Upload Instructions ===
st.markdown("Upload a short **cough recording** in `.wav` format to receive an instant health screening.")

# === File Upload ===
uploaded_file = st.file_uploader("📁 Upload your cough.wav file", type=["wav"])

# === Load Trained Model ===
model_path = "C:/Users/Rita/Desktop/DSA/vitascan_model.pkl"
if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please ensure 'vitascan_model.pkl' is in the correct directory.")
    st.stop()

model = joblib.load(model_path)

# === Feature Extraction Function ===
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) == 0:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        return np.append(np.mean(mfcc.T, axis=0), [zcr, centroid])
    except Exception as e:
        st.error(f"⚠️ Feature extraction failed: {e}")
        return None

# === Run Prediction ===
if uploaded_file is not None:
    temp_path = "temp_cough.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(temp_path, format="audio/wav")

    features = extract_features(temp_path)
    if features is None:
        st.error("❌ Could not process the audio. Please upload a clearer `.wav` cough file.")
    else:
        try:
            X_input = features.reshape(1, -1)
            prediction = model.predict(X_input)[0]
            probas = model.predict_proba(X_input)[0]

            st.success(f"🩺 Prediction: **{prediction.upper()}**")
            st.markdown("### 📊 Model Confidence")
            for label, prob in zip(model.classes_, probas):
                st.write(f"- **{label.capitalize()}**: `{prob*100:.2f}%`")
            st.progress(int(probas[np.argmax(probas)] * 100))
        except Exception as e:
            st.error(f"⚠️ Model prediction failed: {e}")

    os.remove(temp_path)

# === Roadmap / What’s Next Section ===
st.markdown("---")
st.markdown("## 🛠️ What’s Next for VitaScan™")
st.markdown("""
- ✅ Add real-time cough classification with deep learning (CNN/LSTM)
- 🌍 Deploy on Streamlit Cloud for public access
- 📱 Create a mobile-friendly version
- 🧪 Expand dataset for broader conditions (e.g. TB, asthma, Parkinson’s)
- 🔒 Add basic authentication for private use
""")

# === Footer ===
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Made with ❤️ by <strong>Ritalee Monde</strong><br>VitaScan™ 2025 • AI for Public Health</p>",
    unsafe_allow_html=True
)
