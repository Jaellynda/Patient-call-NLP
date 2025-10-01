import os
import sys
import pandas as pd
import nltk
import re
from collections import Counter

# -----------------------------
# Offline-safe NLTK setup
# -----------------------------
# Path to local nltk_data in repo
LOCAL_NLTK_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.insert(0, LOCAL_NLTK_DIR)  # Insert at front so it is searched first

# Verify punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    sys.exit(
        "NLTK resource 'punkt' not found in local folder. "
        "Ensure 'nltk_data/tokenizers/punkt/english.pickle' exists."
    )

# -----------------------------
# Load dataset
# -----------------------------
csv_path = os.path.join(os.path.dirname(__file__), "data", "sample_calls.csv")
if not os.path.exists(csv_path):
    sys.exit(f"CSV file not found at {csv_path}")

df = pd.read_csv(csv_path)

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

# Flatten all tokens
all_tokens = [token for tokens in df["tokens"] for token in tokens]

# Most common words
common_words = Counter(all_tokens).most_common(10)
print("=== Most common words across transcripts ===")
for word, freq in common_words:
    print(f"{word}: {freq}")



