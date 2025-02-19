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

