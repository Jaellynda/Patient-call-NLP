import pandas as pd
import nltk
import re
from collections import Counter
import os



# Use local nltk_data folder
LOCAL_NLTK_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(LOCAL_NLTK_DIR)

# Check if punkt exists
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    raise RuntimeError(
        "NLTK 'punkt' not found in local folder. "
        "Please download it manually as explained."
    )
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

df["tokens"] = df["transcript"].apply(preprocess_text)

# Flatten all tokens
all_tokens = [token for tokens in df["tokens"] for token in tokens]

# Most common words
common_words = Counter(all_tokens).most_common(10)
print("=== Most common words across transcripts ===")
for word, freq in common_words:
    print(f"{word}: {freq}")


