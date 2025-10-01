# Patient-Call-NLP

## 📌 Project Overview
This project builds an **NLP pipeline** to analyze synthetic patient call records.  
It extracts common reasons for calls (e.g., appointments, prescriptions, symptoms), clusters them into topics,  
and summarizes top patient concerns with LLMs.

## ⚙️ What this project demonstrates
- Preprocessing text call records
- Named Entity Recognition (NER) for medical-related terms
- Topic modeling (LDA / BERTopic)
- Summarization using a transformer model (e.g., BART or GPT-style LLM)
- Optional: Expose results via a FastAPI endpoint

## 🚀 Quick Start (local)
1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/Patient-call-NLP.git
   cd Patient-call-NLP
