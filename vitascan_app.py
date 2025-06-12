{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "6e48baa0-9373-4f28-82b0-621eedbd3317",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-11 16:48:24.770 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Rita\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import librosa\n",
    "import joblib\n",
    "import os\n",
    "\n",
    "# Load trained model\n",
    "model = joblib.load(r\"C:\\Users\\Rita\\Desktop\\DSA\\vitascan_model.pkl\")\n",
    "\n",
    "# Title and description\n",
    "st.title(\"🧠 VitaScan™ – Cough Audio Screening\")\n",
    "st.markdown(\"Upload a **cough recording** (`.wav`) to screen for possible respiratory illness.\")\n",
    "\n",
    "# Upload audio file\n",
    "uploaded_file = st.file_uploader(\"Upload your cough.wav file\", type=[\"wav\"])\n",
    "\n",
    "# Feature extraction\n",
    "def extract_features(file_path):\n",
    "    try:\n",
    "        y, sr = librosa.load(file_path, sr=16000)\n",
    "        if len(y) == 0:\n",
    "            return None\n",
    "        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)\n",
    "        zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])\n",
    "        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])\n",
    "        return np.append(np.mean(mfcc.T, axis=0), [zcr, centroid])\n",
    "    except:\n",
    "        return None\n",
    "\n",
    "# Predict on upload\n",
    "if uploaded_file is not None:\n",
    "    # Save file temporarily\n",
    "    temp_path = \"temp_cough.wav\"\n",
    "    with open(temp_path, \"wb\") as f:\n",
    "        f.write(uploaded_file.getbuffer())\n",
    "\n",
    "    # Extract features and predict\n",
    "    features = extract_features(temp_path)\n",
    "    if features is None:\n",
    "        st.error(\"❌ Could not process audio. Try a different recording.\")\n",
    "    else:\n",
    "        X_input = features.reshape(1, -1)\n",
    "        prediction = model.predict(X_input)[0]\n",
    "        probas = model.predict_proba(X_input)[0]\n",
    "\n",
    "        st.success(f\"🩺 Prediction: **{prediction.upper()}**\")\n",
    "        st.write(\"📊 Confidence:\")\n",
    "        st.write({label: f\"{prob*100:.2f}%\" for label, prob in zip(model.classes_, probas)})\n",
    "\n",
    "    # Cleanup\n",
    "    os.remove(temp_path)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "d5468165-a653-40a5-b77c-fa471934a2cc",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
