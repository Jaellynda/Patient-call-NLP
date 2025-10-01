from fastapi import FastAPI
import pandas as pd
from transformers import pipeline
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Patient Call NLP API")

# Load dataset
df = pd.read_csv("data/sample_calls.csv")
documents = df["transcript"].tolist()

# Summarizer (can still run online or offline if model is downloaded)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# BERTopic for topics
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=embedding_model)
topics, probs = topic_model.fit_transform(documents)

@app.get("/")
def root():
    return {"message": "Welcome to the Patient Call NLP API"}

@app.get("/summarize/{call_id}")
def summarize_call(call_id: int):
    row = df[df["call_id"] == call_id]
    if row.empty:
        return {"error": "Call ID not found"}
    text = row.iloc[0]["transcript"]
    summary = summarizer(text, max_length=40, min_length=10, do_sample=False)[0]["summary_text"]
    return {"call_id": int(call_id), "original": text, "summary": summary}

@app.get("/topics")
def get_topics():
    topic_info = topic_model.get_topic_info()
    topics_list = topic_info.to_dict(orient="records")
    return {"topics": topics_list}

