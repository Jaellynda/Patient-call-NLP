import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("data/sample_calls.csv")
documents = df["transcript"].tolist()

print("=== Running BERTopic on patient call transcripts ===")

# Load a small sentence transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize BERTopic
topic_model = BERTopic(embedding_model=embedding_model)

# Fit model
topics, probs = topic_model.fit_transform(documents)

# Print topics
print("=== Topics discovered ===")
topic_info = topic_model.get_topic_info()
print(topic_info)

# Save topics to CSV
topic_info.to_csv("data/discovered_topics.csv", index=False)
print("\nTopics saved to data/discovered_topics.csv")
