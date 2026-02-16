print("loading the inference.py file...")
import re
import json
import contractions
import unicodedata
from bs4 import BeautifulSoup
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


# -------------------------------
# Utility: Text Preprocessing
# -------------------------------
def strip_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text


def remove_accented_chars(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def preprocess_text(text):
    text = strip_html_tags(text)
    text = text.translate(str.maketrans("\n\t\r", "   "))
    text = text.lower()
    text = remove_accented_chars(text)
    text = contractions.fix(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I | re.A)
    text = re.sub(' +', ' ', text)
    return text.strip()


# -------------------------------
# Model & Tokenizer Loading
# -------------------------------
def load_artifacts(model_path=None, tokenizer_path=None, max_sequence_length=1000):
    """
    Loads model and tokenizer for inference.
    """

    # Dynamically build absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(base_dir, "model", "cnn_sentiment_v3.keras")
    if tokenizer_path is None:
        tokenizer_path = os.path.join(base_dir, "model", "tokenizer.json")

    print(" Loading trained model...")
    model = load_model(model_path)

    print(" Loading tokenizer...")
    
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

    return model, tokenizer, max_sequence_length



# -------------------------------
# Prediction Function
# -------------------------------
def predict_sentiment(raw_text, model, tokenizer, max_sequence_length=1000):
    """
    Predict sentiment ('positive' or 'negative') for a raw text.
    """
    clean_text = preprocess_text(raw_text)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=max_sequence_length)

    prob = model.predict(padded, verbose=0)[0][0]
    sentiment = "positive" if prob > 0.5 else "negative"

    return sentiment, float(prob)

print("completed app/inference.py ")