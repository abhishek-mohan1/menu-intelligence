import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Build stopwords
base_stopwords = set(stopwords.words('english'))
custom_stopwords = {
    'good', 'nice', 'great', 'amazing', 'awesome', 'bad', 'best',
    'place', 'restaurant', 'food', 'taste', 'service', 'ambience',
    'ambiance', 'staff', 'experience', 'also', 'really', 'much',
    'definitely', 'absolutely', 'order', 'time', 'visit',
    'well', 'not', 'one', 'get', 'make', 'try', 'like',
    'take', 'serve', 'thank'
}
stop_words = base_stopwords.union(custom_stopwords)
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """Basic text cleaning"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lemmatize_text(text):
    """Lemmatize and remove stopwords using NLTK only"""
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)


def preprocess(df):
    """
    Full preprocessing pipeline.
    Expects a dataframe with 'date' and 'text' columns.
    """
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Auto detect review column name
    possible_names = ['text', 'review', 'reviews', 'review_text', 'comment', 'comments']
    review_col = None
    for name in possible_names:
        if name in df.columns:
            review_col = name
            break

    if review_col is None:
        raise ValueError(
            f"Could not find review column. "
            f"Your file has these columns: {df.columns.tolist()}. "
            f"Please rename your review column to 'text' or 'review'."
        )

    df = df.rename(columns={review_col: 'text'})
    df = df.dropna(subset=['text', 'date'])
    df = df.drop_duplicates(subset=['text'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['clean_review']    = df['text'].apply(clean_text)
    df['processed_text']  = df['clean_review'].apply(lemmatize_text)

    return df