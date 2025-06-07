🛡️ Hate Speech Detection
This is a machine learning project that detects hate speech in text using Natural Language Processing (NLP). The goal is to identify toxic, obscene, threatening, or hateful content in user comments.

🔧 Tools & Technologies
Python

Pandas, NumPy

NLTK, spaCy (for text preprocessing)

Scikit-learn

Tkinter 

📁 Dataset
Source: Hate Speech and Offensive Language Detection
https://www.kaggle.com/code/kirollosashraf/hate-speech-and-offensive-language-detection/input

The dataset contains user comments with labels like:

count,hate_speech_count,offensive_language_count,neither_count,class,tweet

🚀 Steps Involved
Load and clean the dataset

Preprocess the text (lowercase, remove stopwords, lemmatize)

Convert text to numerical format using TF-IDF

Train ML models like Logistic Regression, Random Forest

Evaluate performance using metrics like accuracy and F1-score

▶️ How to Run
Clone the repository

Install required libraries

Open the Jupyter notebook

Run all cells to see model training and predictions

Run hate_speech_gui.py

**Sample Usage**

_Neutral_
1. "I’m going to the store to buy some groceries."

2. "The weather today is pleasant and sunny."

3. "She enjoys reading books in her free time."

_Hate Speech_
1. "Those people don’t belong here and should be kicked out."

2. "I can’t stand that community — they ruin everything."

3. "They are all criminals just because of their race."

_Offensive_
1. "That was the dumbest shit I've ever seen."
2. "Nobody cares about your opinion — it's just shit."
3. "This app is a piece of shit and never works right."

📌 Future Ideas
->Integrate with a Flask/Streamlit web app

->Real-time tweet classification using Twitter API

->Support for multilingual hate speech

->Fine-tuning with BERT-based models

->Classify live tweets

👨‍💻 Author:
Sri Charan Kolachalama
sricharankolachlama@gmail.com
