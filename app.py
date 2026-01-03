import streamlit as st
import joblib
import os

st.write("SERVER FILES:", os.listdir("."))


st.set_page_config(page_title="Spam Classifier")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

st.title("Spam Message Classifier")

msg = st.text_area("Enter message")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message")
    else:
        X = vectorizer.transform([msg])
        result = model.predict(X)[0]

        if result == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")
