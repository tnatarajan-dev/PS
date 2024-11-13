import openai
import numpy as np
from sklearn.cluster import KMeans
import os

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or directly set your key here as a string

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Retrieves the embedding for a single text input using OpenAI's text-embedding-ada-002 model.

    Parameters:
    - text: String, the text to embed.

    Returns:
    - A numpy array containing the embedding for the text.
    """
    response = openai.Embedding.create(input=text, model=model)
    embedding = response['data'][0]['embedding']
    return np.array(embedding)

def vectorize_texts(texts):
    """
    Vectorizes a list of text data.

    Parameters:
    - texts: List of strings to vectorize.

    Returns:
    - A numpy array of embeddings.
    """
    embeddings = [get_embedding(text) for text in texts]
    return np.array(embeddings)

def segment_data(embeddings, n_clusters=3):
    """
    Segments embeddings using KMeans clustering.

    Parameters:
    - embeddings: numpy array of embeddings.
    - n_clusters: Number of clusters to form.

    Returns:
    - Labels indicating the cluster for each embedding.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(embeddings)
    return labels

# Sample text data
texts = [
    "I love the new app interface!",
    "The customer service was very helpful.",
    "The checkout process is confusing.",
    "Great product quality and fast delivery!",
    "Very user-friendly website",
    "The prices are quite high compared to competitors."
]

# Step 1: Generate embeddings for the texts
embeddings = vectorize_texts(texts)

# Step 2: Segment the data into clusters
n_clusters = 3  # Choose the number of clusters you want
labels = segment_data(embeddings, n_clusters=n_clusters)

# Display each text with its cluster label
for i, (text, label) in enumerate(zip(texts, labels)):
    print(f"Text {i+1}: '{text}'")
    print(f"Cluster: {label}")
    print("----")
