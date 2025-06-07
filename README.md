🛡️ Hate Speech Detection using NLP and Machine Learning
This project aims to detect and classify hate speech in text using Natural Language Processing (NLP) and supervised machine learning techniques. It can be used to moderate content on social media platforms, forums, or any online platform to flag toxic content and prevent online harm.

📌 Features
Text classification into categories: toxic, obscene, threat, insult, identity hate

Preprocessing using spaCy and NLTK

Vectorization using TF-IDF

Classification using models like Logistic Regression, Random Forest, and XGBoost

Evaluation with accuracy, precision, recall, and F1-score

Option to predict hate levels for custom input text

🧠 Tech Stack
Language: Python 3.x

Libraries:

NLP: spaCy, nltk, re

Data Handling: pandas, numpy

ML Models: scikit-learn, xgboost

Visualization: matplotlib, seaborn

📂 Dataset
Source: Jigsaw Toxic Comment Classification Challenge - Kaggle

Format: CSV

Columns:

id

comment_text

toxic, severe_toxic, obscene, threat, insult, identity_hate (target labels)

🧹 Preprocessing Pipeline
Lowercasing

Removing special characters and numbers

Tokenization

Lemmatization (using spaCy)

Stopword removal (using NLTK)

🚀 How to Run
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/your-username/hate-speech-detection.git
cd hate-speech-detection
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Jupyter Notebook
bash
Copy
Edit
jupyter notebook hatespeech_detection.ipynb
