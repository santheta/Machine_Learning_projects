import streamlit as st
import pickle

# Load files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Email Spam Classifier")

email = st.text_area("Enter the email text")

if st.button("Predict"):
    if email.strip() == "":
        st.warning("Please enter some text")
    else:
        vector_input = vectorizer.transform([email])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Not Spam")
