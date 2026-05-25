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

app = Flask(__name__, static_folder=".", static_url_path="")
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
    "LLM_Patience",
]

FEATURE_LABELS = [
    "Car Model",
    "Battery Type",
    "Temperature",
    "Daily Commute",
    "Home Charging",
    "Patience Level",
]

FEATURE_OPTIONS = {
    "LLM_Brand": ['Tesla', 'VW', 'Hyundai', 'Ford', 'Nissan', 'Kia', 'GM', 'Polestar', 'Rivian', 'BMW', 'Other'],
    "LLM_Battery": ['NMC', 'LFP'],
    "LLM_Climate": ['Hot', 'Cold', 'Mild'],
    "LLM_Commute": ['Long', 'Short'],
    "LLM_Home_Charging": ['Yes', 'No'],
    "LLM_Patience": ['Low', 'High', 'Unknown']
}

CLASS_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

MODEL_INFO = {
    "model_name":  "Random Forest",
    "metric1_label": "Test Accuracy",
    "metric1_value": "66.1%",
    "metric2_label": "F1 Score (Weighted)",
    "metric2_value": "0.666",
    "training_note": "Trained on EV Reddit Reviews dataset (n=870, 5-fold CV)",
}
# ================================================================
#  END OF STUDENT CONFIG
# ================================================================

# --- Load original training data for dynamically computed insights ---
def extract_brand(model_name):
    if pd.isnull(model_name) or model_name == 'Unknown' or model_name == 'nan':
        return 'Other'
    model_name = str(model_name).strip().lower()
    if 'tesla' in model_name or 'model' in model_name:
        return 'Tesla'
    elif 'vw' in model_name or 'id.' in model_name or 'volkswagen' in model_name:
        return 'VW'
    elif 'hyundai' in model_name or 'ioniq' in model_name or 'kona' in model_name:
        return 'Hyundai'
    elif 'ford' in model_name or 'mach-e' in model_name or 'lightning' in model_name:
        return 'Ford'
    elif 'nissan' in model_name or 'leaf' in model_name or 'ariya' in model_name:
        return 'Nissan'
    elif 'kia' in model_name or 'ev6' in model_name or 'ev9' in model_name:
        return 'Kia'
    elif 'gm' in model_name or 'bolt' in model_name or 'chevy' in model_name or 'chevrolet' in model_name:
        return 'GM'
    elif 'polestar' in model_name:
        return 'Polestar'
    elif 'rivian' in model_name or 'r1t' in model_name or 'r1s' in model_name:
        return 'Rivian'
    elif 'bmw' in model_name or 'i4' in model_name or 'ix' in model_name:
        return 'BMW'
    else:
        return 'Other'

def get_subset_stats(sub_df):
    if sub_df is None or len(sub_df) == 0:
        return {'total': 0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    vc = sub_df['LLM_Satisfaction'].value_counts(normalize=True)
    return {
        'total': int(len(sub_df)),
        'pos': round(float(vc.get('Positive', 0.0)) * 100, 1),
        'neu': round(float(vc.get('Neutral', 0.0)) * 100, 1),
        'neg': round(float(vc.get('Negative', 0.0)) * 100, 1)
    }

df_path = "data_will_use/reddit_training_data_roberta_final_new.csv"
try:
    df_data = pd.read_csv(df_path)
    df_data['LLM_Brand'] = df_data['LLM_Model'].apply(extract_brand)
    print(f"[demo_server] Loaded training dataset with brand mapping: {df_data.shape}")
except Exception as e:
    df_data = None
    print(f"[demo_server] WARNING: Could not load training dataset: {e}")

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
    expected_order = ['LLM_Battery', 'LLM_Climate', 'LLM_Commute', 'LLM_Home_Charging', 'LLM_Brand', 'LLM_Patience']
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

            # Calculate dynamic insights from real training data
            insights = {}
            if df_data is not None:
                try:
                    # 1. Patience
                    p_val = raw.get("LLM_Patience", "Unknown")
                    p_sub = df_data[df_data['LLM_Patience'] == p_val]
                    insights["patience"] = {
                        "value": p_val,
                        "stats": get_subset_stats(p_sub)
                    }

                    # 2. Battery & Climate
                    b_val = raw.get("LLM_Battery", "Unknown")
                    cl_val = raw.get("LLM_Climate", "Unknown")
                    bc_sub = df_data[(df_data['LLM_Battery'] == b_val) & (df_data['LLM_Climate'] == cl_val)]
                    insights["battery_climate"] = {
                        "battery": b_val,
                        "climate": cl_val,
                        "stats": get_subset_stats(bc_sub)
                    }

                    # 3. Charging & Patience
                    ch_val = raw.get("LLM_Home_Charging", "Unknown")
                    chp_sub = df_data[(df_data['LLM_Home_Charging'] == ch_val) & (df_data['LLM_Patience'] == p_val)]
                    insights["charging_patience"] = {
                        "home_charging": ch_val,
                        "patience": p_val,
                        "stats": get_subset_stats(chp_sub)
                    }

                    # 4. Brand
                    br_val = raw.get("LLM_Brand", "Other")
                    br_sub = df_data[df_data['LLM_Brand'] == br_val]
                    insights["brand"] = {
                        "brand": br_val,
                        "stats": get_subset_stats(br_sub)
                    }
                except Exception as ex:
                    print(f"[demo_server] Error computing insights: {ex}")

            result = {
                "predicted_class": class_name,
                "predicted_class_int": pred_class_int,
                "confidence": confidence,
                "all_probabilities": all_proba,
                "data_insights": insights
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
