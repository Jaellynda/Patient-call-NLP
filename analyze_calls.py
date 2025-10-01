import pandas as pd
import nltk
import re
from collections import Counter
import os

# -----------------------------
# Make sure NLTK 'punkt' is available offline
# -----------------------------
# Set a local nltk_data folder
LOCAL_NLTK_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(LOCAL_NLTK_DIR, exist_ok=True)

# Add it to nltk data paths
nltk.data.path.append(LOCAL_NLTK_DIR)

# Try to find 'punkt', download to local folder if not found
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading 'punkt' to local nltk_data folder...")
    nltk.download('punkt', download_dir=LOCAL_NLTK_DIR)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/sample_calls.csv")

print("=== Patient Call Dataset ===")
print(f"Total calls: {len(df)}")
print(df.head(), "\n")

# -----------------------------
# Function to clean and tokenize text
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = nltk.word_tokenize(text)
    return tokens

# Apply preprocessing
df["tokens"] = df["transcript"].apply(preprocess_text)

# Flatten all tokens across calls
all_tokens = [token for tokens in df["tokens"] for token in tokens]

# Show most common words
common_words = Counter(all_tokens).most_common(10)

print("=== Most common words across transcripts ===")
for word, freq in common_words:
    print(f"{word}: {freq}")

