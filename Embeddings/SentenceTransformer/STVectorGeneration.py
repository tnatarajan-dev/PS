from sentence_transformers import SentenceTransformer

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # This is a smaller, faster model from Sentence Transformers

# Define a list of customer descriptions
customer_descriptions = [
    "John is a 30-year-old software engineer from New York interested in technology and fitness.",
    "Sarah is a 45-year-old teacher from Chicago who loves traveling and cooking.",
    "Michael is a 28-year-old data scientist from San Francisco with a passion for music and art."
]

# Generate embeddings for each description
embeddings = model.encode(customer_descriptions)

# Convert embeddings to a dictionary format for easy access
customer_embeddings = {f"Customer_{i+1}": embeddings[i] for i in range(len(customer_descriptions))}

# Display the embeddings
for customer, embedding in customer_embeddings.items():
    #print(f"{customer} embedding: {embedding[:5]}...")  # Display the first 5 values for brevity
    print(f"{customer} embedding: {embedding}")  # Display the first 5 values for brevity

# Write embbeddings to text files
vector_filename = "vector_output.txt"
with open(vector_filename, 'w') as f:
    for customer, embedding in customer_embeddings.items():
        f.write('{}:{}\n'.format(customer, ' '.join(['{:e}'.format(item) for item in embedding])))

