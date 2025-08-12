import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Breast Cancer Classification — Inference Only", layout="centered")
st.title("Breast Cancer Classification (Inference Only)")
st.caption("This app does not train a model. It only loads a pre-trained scikit-learn Pipeline and makes predictions.")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file at: {MODEL_PATH}. Place your pre-trained Pipeline as 'model.pkl' next to app.py.")
    return joblib.load(MODEL_PATH)

model = load_model()
st.success("Model loaded successfully.")

FEATURE_NAMES = [ 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

def predict_df(df: pd.DataFrame):
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")
    X = df[FEATURE_NAMES]
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    pred = model.predict(X)
    return pred, proba

tab_csv, tab_manual = st.tabs(["📄 CSV Upload", "⌨️ Manual Input"])
tmp = pd.read_csv('data.csv')
with tab_csv:
    st.write("Upload a CSV whose columns match the exact feature names (see template).")
    if st.button("Download CSV Template"):
        tmp = pd.DataFrame([{f: 0.0 for f in FEATURE_NAMES}])
    st.download_button("Save template.csv", data=tmp.to_csv(index=False).encode("utf-8"),
                        file_name="template.csv", mime="text/csv")
    up = st.file_uploader("CSV file", type=["csv"], key="csv")
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            st.write("Preview:", df_in.head())
            if st.button("Predict from CSV"):
                pred, proba = predict_df(df_in)
                out = df_in.copy()
                out["prediction"] = ["Benign" if p==1 else "Malignant" for p in pred]
                if proba is not None and proba.shape[1] == 2:
                    out["prob_malignant"] = proba[:, 0]
                    out["prob_benign"]= proba[:, 1]
                st.subheader("Results")
                st.dataframe(out, use_container_width=True)
                st.download_button("Download results.csv", data=out.to_csv(index=False).encode("utf-8"),
                                   file_name="results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"CSV error: {e}")

with tab_manual:
    st.write("Enter feature values manually (defaults are 0.0).")
    vals = {}
    for f in FEATURE_NAMES:
        vals[f] = st.number_input(f, value=0.0, format="%.4f", key=f"num_{f}")
    if st.button("Predict single row"):
        try:
            df_one = pd.DataFrame([vals])
            pred, proba = predict_df(df_one)
            label = "Benign" if int(pred[0]) == 1 else "Malignant"
            st.write("Prediction:", label)
            if proba is not None and proba.shape[1] == 2:
                st.write("Probability (malignant, benign):",
                         float(proba[0][0]), float(proba[0][1]))
        except Exception as e:
            st.error(f"Inference error: {e}")