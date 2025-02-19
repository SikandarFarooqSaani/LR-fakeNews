import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
with open("fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit App UI
st.title("📰 Fake News Detector")
st.subheader("Enter a news article below to check if it's real or fake.")
# Project Information
st.sidebar.title("📌 Project Details")
st.sidebar.markdown("**Project:** Fake News Detector")  
st.sidebar.markdown("**Institute:** University of Management and Technology")  
st.sidebar.markdown("**Advisor:** Muhammad Asim Butt")  
st.sidebar.markdown("**Group Members:**")  
st.sidebar.markdown("- Sikandar Farooq")  
st.sidebar.markdown("- Musfira Rabia Ramzan")  
st.sidebar.markdown("- Abdullah Rehman")  
st.sidebar.markdown("- Mariyum Iftikhar")  

# User Input
user_input = st.text_area("Paste the news content here:", "")

# Prediction Function
def predict_news(text):
    transformed_text = vectorizer.transform([text])  # Convert text to TF-IDF format
    prediction = model.predict(transformed_text)  # Predict using trained model
    return "✅ Real News" if prediction[0] == 1 else "❌ Fake News"

# Prediction Button
if st.button("Check News"):
    if user_input.strip():
        result = predict_news(user_input)
        st.subheader(f"Prediction: {result}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# About Section
st.markdown("---")
st.markdown("🔍 **How it Works:** This model uses Logistic Regression with TF-IDF vectorization to classify news articles as real or fake.")
st.markdown("📌 **Built with:** Python, Scikit-learn, Streamlit")
st.markdown("🛠 **Trained on:** Kaggle Fake News Dataset")

