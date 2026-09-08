# ============================================================
# HEART DISEASE PREDICTION - MODEL TRAINING (RUN THIS IN COLAB)
# ============================================================
# This trains the Random Forest model (best performer in the
# notebook) and saves it so you can load it in Streamlit.

# ---- 1. Install/Import ----
!pip install scikit-learn pandas numpy joblib -q

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ---- 2. Upload heart.csv ----
from google.colab import files
uploaded = files.upload()   # choose heart.csv when prompted

dataset = pd.read_csv("heart.csv")
print(dataset.shape)
dataset.head()

# ---- 3. Train / test split ----
predictors = dataset.drop("target", axis=1)
target = dataset["target"]

X_train, X_test, Y_train, Y_test = train_test_split(
    predictors, target, test_size=0.20, random_state=0
)

# ---- 4. Find the best random_state for Random Forest (same idea as notebook) ----
max_accuracy = 0
best_x = 0
for x in range(200):          # 200 is plenty; 2000 in the notebook was overkill
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

print(f"Best random_state: {best_x}  |  Accuracy: {max_accuracy}%")

# ---- 5. Train final model on ALL data with the best random_state ----
final_model = RandomForestClassifier(random_state=best_x)
final_model.fit(predictors, target)   # train on full dataset for deployment

# ---- 6. Save the model + the column order (needed by Streamlit app) ----
joblib.dump(final_model, "heart_disease_model.pkl")
joblib.dump(list(predictors.columns), "model_columns.pkl")

print("Saved: heart_disease_model.pkl and model_columns.pkl")

# ---- 7. Download the files to your computer ----
files.download("heart_disease_model.pkl")
files.download("model_columns.pkl")

# ============================================================
# Next step: put heart_disease_model.pkl and model_columns.pkl
# in the SAME folder as app.py, then run:
#   streamlit run app.py
# ============================================================
