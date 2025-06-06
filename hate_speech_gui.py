import customtkinter as ctk
import joblib
import re
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk

nltk.download('wordnet')

# Appearance settings
ctk.set_appearance_mode("dark")  # Options: "Light", "Dark", "System"
ctk.set_default_color_theme("blue")

# Preprocessing steps
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

def preprocess(text):
    text = clean_text(text)
    text = fast_text_processing(text)
    text = lemmatize_text_nltk(text)
    return text

# Load model and vectorizer
model = joblib.load('hate_speech_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

label_map = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}

# Prediction logic
def predict():
    text = entry.get()
    if not text.strip():
        result_label.configure(text="⚠️ Please enter a statement.", text_color="orange")
        return
    cleaned_text = preprocess(text)
    vec = vectorizer.transform([cleaned_text])
    prediction_array = model.predict(vec)[0]  # E.g., [0, 1, 0]
    prediction_num = prediction_array.argmax()
    prediction_label = label_map.get(prediction_num, "Unknown")
    result_label.configure(text=f"🔍 Prediction: {prediction_label}", text_color="lightgreen")

# GUI Setup
root = ctk.CTk()
root.title("🛡️ Hate Speech Detector")
root.geometry("500x300")

title_label = ctk.CTkLabel(root, text="Hate Speech Detection", font=ctk.CTkFont(size=20, weight="bold"))
title_label.pack(pady=15)

entry = ctk.CTkEntry(root, width=400, placeholder_text="Type a sentence to analyze...")
entry.pack(pady=15)

analyze_button = ctk.CTkButton(root, text="Analyze", command=predict)
analyze_button.pack(pady=10)

result_label = ctk.CTkLabel(root, text="Prediction: ", font=ctk.CTkFont(size=14))
result_label.pack(pady=20)

root.mainloop()
