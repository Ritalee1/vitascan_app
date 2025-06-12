import streamlit as st
import numpy as np
import librosa
import joblib
import os

# Load trained model
model = joblib.load("C:/Users/Rita/Desktop/DSA/vitascan_model.pkl")

# Title and description
st.title("🧠 VitaScan™ – Cough Audio Screening")
st.markdown("Upload a **cough recording** (`.wav`) to screen for possible respiratory illness.")

# Upload audio file
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
    except:
        return None

# Predict on upload
if uploaded_file is not None:
    # Save file temporarily
    temp_path = "temp_cough.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features and predict
    features = extract_features(temp_path)
    if features is None:
        st.error("❌ Could not process audio. Try a different recording.")
    else:
        X_input = features.reshape(1, -1)
        prediction = model.predict(X_input)[0]
        probas = model.predict_proba(X_input)[0]

        st.success(f"🩺 Prediction: **{prediction.upper()}**")
        st.write("📊 Confidence:")
        st.write({label: f"{prob*100:.2f}%" for label, prob in zip(model.classes_, probas)})

    # Cleanup
    os.remove(temp_path)
