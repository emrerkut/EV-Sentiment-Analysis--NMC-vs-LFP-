"""
========================================================
  ML MODEL DEMO — BACKEND SERVER
  Grup: [Grup Adınız / Group Name]
  Proje: [Proje Başlığı / Project Title]
========================================================
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__, static_folder=".")
CORS(app)  # allows the HTML file to call this server

# ================================================================
#  STUDENT CONFIG — edit this section before your presentation
# ================================================================

MODEL_PATH = "ev_sentiment_model.joblib"          
SCALER_PATH = None                   
MODEL_TYPE = "classification"            

FEATURE_NAMES = [
    "LLM_Brand",
    "LLM_Battery",
    "LLM_Climate",
    "LLM_Commute",
    "LLM_Home_Charging",
]

FEATURE_LABELS = [
    "Car Model",
    "Battery Type",
    "Temperature",
    "Daily Commute",
    "Home Charging",
]

FEATURE_OPTIONS = {
    "LLM_Brand": ['Audi', 'BMW', 'BYD', 'Dodge', 'Dongfeng', 'F150', 'Ford', 'GM', 'Genesis', 'Honda', 'Hyundai', 'Ioniq', 'Kia', 'Lucid', 'MG4', 'Mercedes', 'Nio', 'Nissan', 'Polestar', 'Porsche', 'Renault', 'Rivian', 'Spark', 'Subaru', 'Tesla', 'Toyota', 'VW', 'Vauxhall', 'Volvo', 'Xiaomi', 'Zoe', 'peugeot'],
    "LLM_Battery": ['NMC', 'LFP'],
    "LLM_Climate": ['Hot', 'Cold', 'Mild'],
    "LLM_Commute": ['Long', 'Short'],
    "LLM_Home_Charging": ['Yes', 'No']
}

CLASS_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

MODEL_INFO = {
    "model_name":  "KNN",
    "metric1_label": "Test Accuracy",
    "metric1_value": "47%",
    "metric2_label": "Test Accuracy and Confusion Matrix",
    "metric2_value": "0.47",
    "training_note": "Trained on EV Reddit Reviews dataset (n=656)",
}
# ================================================================
#  END OF STUDENT CONFIG
# ================================================================


# --- load model and scaler at startup ---
base_dir = os.path.dirname(os.path.abspath(__file__))
actual_model_path = os.path.join(base_dir, MODEL_PATH)

print(f"\n[demo_server] Loading model from: {actual_model_path}")
def fix_scikit_learn_1_8_0_bug(estimator):
    """Recursively set _fill_dtype on SimpleImputer to fix scikit-learn version mismatch (1.7.x to 1.8.x)."""
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        
        if isinstance(estimator, SimpleImputer) and not hasattr(estimator, '_fill_dtype'):
            estimator._fill_dtype = getattr(estimator, '_fit_dtype', object)
        if isinstance(estimator, Pipeline):
            for name, step in estimator.steps:
                fix_scikit_learn_1_8_0_bug(step)
        if isinstance(estimator, ColumnTransformer):
            for name, transformer, columns in estimator.transformers:
                fix_scikit_learn_1_8_0_bug(transformer)
        if hasattr(estimator, 'get_params'):
            for key, param in estimator.get_params(deep=False).items():
                fix_scikit_learn_1_8_0_bug(param)
    except Exception:
        pass

try:
    model = joblib.load(actual_model_path)
    fix_scikit_learn_1_8_0_bug(model)
    print(f"[demo_server] Model loaded: {type(model).__name__}")
except FileNotFoundError:
    print(f"[demo_server] ERROR: Model file not found at '{actual_model_path}'")
    model = None

scaler = None
if SCALER_PATH:
    actual_scaler_path = os.path.join(base_dir, SCALER_PATH)
    try:
        scaler = joblib.load(actual_scaler_path)
    except FileNotFoundError:
        pass

@app.route("/")
def index():
    """Serve the UI HTML file."""
    return send_from_directory(".", "demo_ui.html")

@app.route("/config")
def get_config():
    """Return model config so the UI can build itself dynamically."""
    return jsonify({
        "model_type":   MODEL_TYPE,
        "feature_names": FEATURE_NAMES,
        "feature_labels": FEATURE_LABELS,
        "feature_options": FEATURE_OPTIONS,
        "class_labels": CLASS_LABELS,
        "model_info":   MODEL_INFO,
        "model_ready":  model is not None,
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Receive feature values and return model prediction."""
    if model is None:
        return jsonify({"error": f"Model not loaded. Check that '{MODEL_PATH}' exists."}), 500

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Request must include a 'features' dict."}), 400

    raw = data["features"]

    # KULLANICI ARAYÜZDEN METİN GİRDİĞİ İÇİN FLOAT YAPMIYORUZ!
    # Direkt string alıp Pandas DataFrame oluşturuyoruz (KNN Pipeline için zorunlu)
    try:
        values = {name: [str(raw[name])] for name in FEATURE_NAMES}
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}. Expected: {FEATURE_NAMES}"}), 400

    X = pd.DataFrame(values)
    
    # Modelin beklediği sütun sırasını zorunlu kıl (fit sırasında görülen sıra)
    expected_order = ['LLM_Battery', 'LLM_Climate', 'LLM_Commute', 'LLM_Home_Charging', 'LLM_Brand']
    X = X[expected_order]

    if scaler is not None:
        X = scaler.transform(X)

    try:
        if MODEL_TYPE == "regression":
            pred = float(model.predict(X)[0])
            result = {"prediction": round(pred, 3)}

        else:  # classification
            pred_class_raw = model.predict(X)[0]
            class_name = str(pred_class_raw)

            # Eger cikti numerikse, isme cevir; stringse aynen kalsin
            pred_class_int = 0
            for k, v in CLASS_LABELS.items():
                if v.lower() == class_name.lower():
                    pred_class_int = k

            confidence = None
            all_proba = {}
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = round(float(max(proba)), 3)
                classes = getattr(model, "classes_", [CLASS_LABELS.get(i, str(i)) for i in range(len(proba))])
                all_proba = {str(c): round(float(p), 3) for c, p in zip(classes, proba)}

            result = {
                "predicted_class": class_name,
                "predicted_class_int": pred_class_int,
                "confidence": confidence,
                "all_probabilities": all_proba,
            }

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify(result)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ML Demo Server")
    print("  Open in browser: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
