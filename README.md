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
