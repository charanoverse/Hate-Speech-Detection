import joblib

# Load saved model and vectorizer
model = joblib.load('hate_speech_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Reuse your preprocessing functions from before:
import re
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|[^a-z\s]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fast_text_processing(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

def lemmatize_text_nltk(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def preprocess_text(text):
    text = clean_text(text)
    text = fast_text_processing(text)
    text = lemmatize_text_nltk(text)
    return text

def predict_text(text):
    processed = preprocess_text(text)
    vect = tfidf.transform([processed])
    prediction = model.predict(vect)
    label_map = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}
    label_index = prediction.argmax()
    return label_map[label_index]

# Example usage:
while True:
    user_input = input("Enter text (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    print("Prediction:", predict_text(user_input))
