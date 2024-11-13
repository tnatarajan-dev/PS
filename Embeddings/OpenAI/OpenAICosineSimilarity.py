import openai
import numpy as np
from scipy.spatial.distance import cosine

# Load the OpenAI API key
openai.api_key = "YOUR_API_KEY"

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]


# Load text from file
with open("file.txt", "r") as f:
    text1 = f.read()

# Get embeddings for the text from file and another text
embedding1 = get_embedding(text1)
embedding2 = get_embedding("Another text to compare")

# Calculate cosine similarity
similarity = 1 - cosine(embedding1, embedding2)
print("Cosine similarity:", similarity)
