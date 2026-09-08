# ❤️ Heart Disease Prediction

A machine learning web app that predicts whether a patient is at risk of heart disease, based on 13 clinical features. Built with **scikit-learn** (Random Forest) and deployed with **Streamlit**.

## 📁 Project Structure

```
Heart-Disease-Prediction/
├── Heart_disease_prediction (1).ipynb   # Full EDA + model comparison notebook
├── train_model_colab.py                 # Script to train & export the final model (run in Colab)
├── app.py                               # Streamlit web app
├── heart (1).csv                        # Dataset
├── heart_disease_model.pkl              # Trained Random Forest model
├── model_columns.pkl                    # Feature column order (used by the app)
├── requirements.txt                     # Python dependencies
└── README.md
```

## 🧠 About the Model

Multiple algorithms were trained and compared in the notebook:

| Algorithm | Accuracy |
|---|---|
| Logistic Regression | ~85% |
| Naive Bayes | ~85% |
| SVM (Linear) | ~82% |
| K-Nearest Neighbors | ~67% |
| Decision Tree | ~79% |
| **Random Forest** | **~90%** ✅ |
| XGBoost | ~85% |
| Neural Network | ~85% |

**Random Forest** gave the best accuracy and was selected as the final deployed model.

### Features used

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0–4) |
| `thal` | Thalassemia type |

## 🛠️ Setup & Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/mirzayasirabdullahbaig07/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. Open the local URL Streamlit shows (usually `http://localhost:8501`).

## 🔁 Retraining the Model

To retrain the model with new data or parameters:

1. Open `train_model_colab.py` in Google Colab.
2. Upload `heart (1).csv` when prompted.
3. Run all cells — this regenerates `heart_disease_model.pkl` and `model_columns.pkl`.
4. Download both files and replace them in this repo (same folder as `app.py`).

## ☁️ Deploying on Streamlit Cloud

1. Push this repo to GitHub (already done ✅).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app**, select this repository and branch.
4. Set the main file path to `app.py`.
5. Click **Deploy**.

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is not a certified medical diagnostic tool and should not be used as a substitute for professional medical advice.

## 👤 Author

**Mirza Yasir Abdullah Baig**
