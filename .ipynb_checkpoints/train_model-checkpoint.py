import pandas as pd
import re
import string
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep required columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Convert labels
df["label_num"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Same text cleaning used in your notebook
stop_words = set(stopwords.words("english"))

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

# Clean messages
df["clean_message"] = df["message"].apply(clean_text)

# Features and labels
X = df["clean_message"]
y = df["label_num"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)

X_train_tfidf = vectorizer.fit_transform(X_train)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save trained model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained successfully!")
print("✅ model.pkl created!")
print("✅ vectorizer.pkl created!")