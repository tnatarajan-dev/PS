import openai

openai.api_key = "YOUR_API_KEY"

def create_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

embedding = create_embedding("This is a sample text to embed.")
print(embedding)
