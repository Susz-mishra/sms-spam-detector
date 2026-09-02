import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords

# Load saved model and TF-IDF vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Stopwords
stop_words = set(stopwords.words("english"))

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    text = " ".join(
        word for word in text.split()
        if word not in stop_words
    )
    return text


# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱"
)

st.title("📱 SMS Spam Detector")
st.write("Enter an SMS message and let the ML model classify it.")

# Message input
message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You have won a prize..."
)

# Prediction
if st.button("🔍 Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(message)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ HAM MESSAGE")