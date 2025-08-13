# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Project", layout="wide")

@st.cache_data
def load_data(path="data/diabetes_dataset.csv"):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path="model.pkl"):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Explorer", "Visualisations", "Predict", "Model Performance"])

data = load_data()
model = load_model()

if page == "Home":
    st.title("Diabetics Prediction Analyzer")
    st.write("check out your own diabetic level")
    st.write("## Dataset overview")
    st.write(f"Shape: {data.shape}")
    st.write(data.describe())

if page == "Data Explorer":
    st.header("Data Explorer")
    if st.checkbox("Show raw data"):
        st.dataframe(data.sample(200))
    # simple filters
    col = st.selectbox("Filter column", options=data.columns)
    if data[col].dtype in ['int64','float64']:
        rng = st.slider("Range", float(data[col].min()), float(data[col].max()), (float(data[col].min()), float(data[col].max())))
        st.dataframe(data[(data[col] >= rng[0]) & (data[col] <= rng[1])])
    else:
        vals = st.multiselect("Values", options=data[col].unique())
        if vals:
            st.dataframe(data[data[col].isin(vals)])

if page == "Visualisations":
    st.header("Visualisations")
    # Example: histogram
    num_cols = data.select_dtypes(['number']).columns.tolist()
    if num_cols:
        col = st.selectbox("Numeric column for histogram", num_cols)
        fig = px.histogram(data, x=col)
        st.plotly_chart(fig, use_container_width=True)
    # Add correlation heatmap
    if st.checkbox("Show correlation heatmap"):
        fig2 = px.imshow(data.corr())
        st.plotly_chart(fig2, use_container_width=True)

if page == "Predict":
    st.header("Make a prediction")
    if model is None:
        st.error("Model not loaded. Check model.pkl")
    else:
        # Build input widgets dynamically for numeric features
        # NOTE: you must ensure the order/columns match training pipeline
        sample = {}
        # Example manual input (replace with your dataset feature names)
        # For demonstration:
        for feat in data.columns.drop('target', errors='ignore')[:6]:  # adjust or hardcode correct features
            if data[feat].dtype in ['int64','float64']:
                sample[feat] = st.number_input(feat, float(data[feat].min()), float(data[feat].max()), float(data[feat].median()))
            else:
                sample[feat] = st.selectbox(feat, options=data[feat].unique())
        if st.button("Predict"):
            try:
                X_new = pd.DataFrame([sample])
                pred = model.predict(X_new)
                st.success(f"Prediction: {pred[0]}")
                # If classifier, show probability if available
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_new)
                    st.write(f"Probabilities: {proba[0]}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

if page == "Model Performance":
    st.header("Model Performance on test set")
    # Load a saved test split or compute here (prefer to store test set)
    if st.button("Evaluate on holdout"):
        # This is illustrative; adapt to your saved test data
        from sklearn.model_selection import train_test_split
        X = data.drop(columns=['Outcome'])
        y = data['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        try:
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.write(f"Accuracy: {acc:.3f}")
            st.text(classification_report(y_test, preds))
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            ax.imshow(cm, interpolation='nearest')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Evaluation error: {e}")
