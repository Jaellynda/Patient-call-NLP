# Patient-Call-NLP

## Project Overview
This project builds an **NLP pipeline** to analyze synthetic patient call records.  
It extracts common reasons for calls (e.g., appointments, prescriptions, symptoms), clusters them into topics,  
and summarizes top patient concerns with LLMs.

## What this project demonstrates
- Preprocessing text call records
- Named Entity Recognition (NER) for medical-related terms
- Topic modeling (LDA / BERTopic)
- Summarization using a transformer model (e.g., BART or GPT-style LLM)
- Optional: Expose results via a FastAPI endpoint

## 
## 🚀 Running the Patient Call NLP API

This project includes a FastAPI service that lets you interact with the NLP pipeline through HTTP endpoints.

### 1️Install dependencies
```bash
pip install -r requirements.txt

---

