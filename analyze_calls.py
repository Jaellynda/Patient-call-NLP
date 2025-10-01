import pandas as pd
import nltk
import re
from collections import Counter
import os
import sys

# -----------------------------
# Offline-safe NLTK setup
# -----------------------------
# Local nltk_data folder in repo
LOCAL_NLTK_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(LOCAL_NLTK_DIR)

# Correct resource name: 'punkt', not 'punkt_tab'
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    sys.exit(
        "NLTK resource 'punkt' not found in local folder. \n"
        "Please manually download it and place it in 'nltk_data/tokenizers/punkt/'"
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


