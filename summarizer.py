import pandas as pd
from transformers import pipeline

# Load dataset
df = pd.read_csv("data/sample_calls.csv")
documents = df["transcript"].tolist()

print("=== Summarizing patient calls ===")

# Load summarization pipeline (BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Summarize first 3 calls for demonstration
for i, doc in enumerate(documents[:3]):
    summary = summarizer(doc, max_length=40, min_length=10, do_sample=False)[0]["summary_text"]
    print(f"\nCall {i+1}:")
    print(f"Original: {doc}")
    print(f"Summary: {summary}")

print("\n✅ Summaries generated for sample calls.")
