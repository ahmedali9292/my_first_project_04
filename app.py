import streamlit as st
import joblib
import numpy as np



st.write("""My First Python App""")



model = joblib.load("MultinomialNB.pkl")   # <-- MUST match filename

st.title("ML Prediction App")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")

if st.button("Predict"):
    result = model.predict([[f1, f2]])
    st.success(f"Prediction: {result[0]}")