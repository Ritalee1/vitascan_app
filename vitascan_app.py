import streamlit as st
import numpy as np
import librosa
import joblib
import os
import soundfile as sf  # For audio preview

# Page config
st.set_page_config(
    page_title="VitaScan™",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

# Load model
model = joblib.load("C:/Users/Rita/Desktop/DSA/vitascan_model.pkl")

# Header
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🧠 VitaScan™</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Cough Audio Screening App</h3>", unsafe_allow_html=True)
st.write("Upload a **cough recording** (.wav) to screen for possible respiratory illness.")

# Upload
uploaded_file = st.file_uploader("Upload your cough.wav file", type=["wav"])

# Feature extraction
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
        return None

# Prediction
if uploaded_file is not None:
    # Save file
    temp_path = "temp_cough.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(temp_path, format="audio/wav")

    features = extract_features(temp_path)
    if features is None:
        st.error("❌ Could not process audio. Try a different recording.")
    else:
        X_input = features.reshape(1, -1)
        prediction = model.predict(X_input)[0]
        probas = model.predict_proba(X_input)[0]

        st.success(f"🩺 Prediction: **{prediction.upper()}**")
        st.progress(int(probas[np.argmax(probas)] * 100))

        st.markdown("### 📊 Confidence")
        for label, prob in zip(model.classes_, probas):
            st.write(f"- **{label.capitalize()}**: `{prob*100:.2f}%`")

    os.remove(temp_path)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Made with ❤️ by <strong>Ritalee Monde</strong> • VitaScan™ 2025</p>",
    unsafe_allow_html=True
)
