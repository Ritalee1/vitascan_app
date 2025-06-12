# VitaScan™ – AI-Powered Cough Screening Tool

VitaScan™ is a lightweight AI web app that allows users to upload a cough recording and receive a rapid health screening prediction. It uses machine learning to analyze audio features and classify the sample as **Healthy** or **Potentially Positive** for respiratory conditions.

Built with:
- 🧠 `scikit-learn` (Random Forest model)
- 🔊 `librosa` (audio feature extraction)
- 🌐 `Streamlit` (web app frontend)

---

## 🩺 Try It Out

> Upload a `.wav` file of a cough, and VitaScan will analyze it and show:
- A health prediction
- Confidence scores for each class

---

## 🚀 Deployment

VitaScan is ready to deploy on [Streamlit Community Cloud](https://streamlit.io/cloud).

To run locally:

```bash
git clone https://github.com/your-username/vitascan-app.git
cd vitascan-app
pip install -r requirements.txt
streamlit run vitascan_app.py

📁 File Structure
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
