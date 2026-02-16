print ("Loading app/data.py...")
import pandas as pd
import numpy as np
import re
import unicodedata
import contractions
import tqdm
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os


# -------------------------------
# Text Cleaning Functions
# -------------------------------
def strip_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text


def remove_accented_chars(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def pre_process_text(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        doc = strip_html_tags(doc)
        doc = doc.translate(str.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = contractions.fix(doc)
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I | re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()
        norm_docs.append(doc)
    return norm_docs


# -------------------------------
# Dataset Preparation
# -------------------------------
def load_dataset(filename="IMDB_Dataset.csv"):
    """Load the IMDB dataset from the data folder."""
    
    # Get the absolute path to the data file
    base_dir = os.path.dirname(os.path.abspath(__file__))   
    data_path = os.path.join(base_dir, "data", filename)    
    
    # Read the datasetS
    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset with shape: {df.shape}")
    return df


def prepare_data(df, vocab_size_limit=None, max_sequence_length=1000):
    """Clean, tokenize, pad, and encode dataset."""
    reviews = df['review'].values
    sentiments = df['sentiment'].values

    # Split train/test
    train_reviews = reviews[:35000]
    test_reviews = reviews[35000:]
    train_sentiments = sentiments[:35000]
    test_sentiments = sentiments[35000:]

    # Clean text
    print("🧹 Cleaning text data...")
    norm_train_reviews = pre_process_text(train_reviews)
    norm_test_reviews = pre_process_text(test_reviews)

    # Tokenization
    print("🔤 Tokenizing text...")
    tokenizer = Tokenizer(oov_token='<UNK>')
    tokenizer.fit_on_texts(norm_train_reviews)
    tokenizer.word_index['<PAD>'] = 0

    train_sequences = tokenizer.texts_to_sequences(norm_train_reviews)
    test_sequences = tokenizer.texts_to_sequences(norm_test_reviews)

    X_train = pad_sequences(train_sequences, maxlen=max_sequence_length)
    X_test = pad_sequences(test_sequences, maxlen=max_sequence_length)

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_sentiments)
    y_test = le.transform(test_sentiments)

    vocab_size = len(tokenizer.word_index)
    print(f"📚 Vocabulary size: {vocab_size}")

    return X_train, X_test, y_train, y_test, tokenizer, vocab_size

print("Finished loading app/data.py.")