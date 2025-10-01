import pandas as pd
import nltk
import re
from collections import Counter

# Make sure NLTK resources are available
nltk.download('punkt', quiet=True)

# Load the dataset
df = pd.read_csv("data/sample_calls.csv")

print("=== Patient Call Dataset ===")
print(f"Total calls: {len(df)}")
print(df.head(), "\n")

# Function to clean and tokenize text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = nltk.word_tokenize(text)
    return tokens

# Apply preprocessing to transcripts
df["tokens"] = df["transcript"].apply(preprocess_text)

# Flatten all tokens across calls
all_tokens = [token for tokens in df["tokens"] for token in tokens]

# Show most common words
common_words = Counter(all_tokens).most_common(10)

print("=== Most common words across transcripts ===")
for word, freq in common_words:
    print(f"{word}: {freq}")
