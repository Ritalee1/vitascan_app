# VitaScan™ – AI-Powered Cough Screening Tool

**VitaScan™** is an AI-based web app that analyzes cough sounds to screen for potential respiratory illness. Users can upload a `.wav` cough recording and receive a prediction: **Healthy** or **Positive**. The goal is to empower early screening in low-resource or remote environments using only voice input.

Built with:
- 🧠 `scikit-learn` (Random Forest model)
- 🔊 `librosa` (audio feature extraction)
- 🌐 `Streamlit` (web app frontend)

---

## 🩺 Try It Out

Upload a short `.wav` cough audio and VitaScan will:
- Extract features like MFCCs, Zero Crossing Rate, and Spectral Centroid
- Predict if the cough is **Healthy** or **Potentially Positive**
- Display class probabilities as confidence levels

---

## 🚀 Run Locally

```bash
git clone https://github.com/your-username/vitascan-app.git
cd vitascan-app
pip install -r requirements.txt
streamlit run vitascan_app.py

## 📁 File Structure
bash
Copy
Edit
vitascan-app/
├── vitascan_app.py          # Streamlit interface
├── vitascan_model.pkl       # Trained Random Forest model
└── requirements.txt         # Python dependencies
📦 Requirements
Install everything with:

bash
Copy
Edit
pip install -r requirements.txt
Required packages:

nginx
Copy
Edit
streamlit
librosa
scikit-learn
joblib
numpy

## 🤖 Model Information
The VitaScan model is trained on audio features from labeled cough datasets. It uses:

MFCCs (Mel-frequency cepstral coefficients) for timbre and tone

ZCR (Zero Crossing Rate) for signal energy

Spectral Centroid for brightness

The model classifies:

healthy

positive (merged from multiple positive subtypes)

Accuracy is best on balanced datasets. Ideal for prototyping and early research use.

## 👩‍💻 Creator
Ritalee Monde
Public Health + AI Innovator
💡 Building AI-powered healthtech solutions for impact.
📫 https://www.linkedin.com/in/ritalee-pamela-monde-990816142

## ⚠️ Disclaimer
This tool is for experimental and educational purposes only.
VitaScan™ is not a medical diagnostic tool and should not be used to replace clinical testing or professional healthcare advice.

## 📜 License
MIT License – Free to use, modify, and distribute.
Please credit the original creator if you build upon this work.
