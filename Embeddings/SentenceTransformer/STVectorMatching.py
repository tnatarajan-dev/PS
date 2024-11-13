from sentence_transformers import SentenceTransformer, util
import numpy as np

def create_vector_dict(filename):
    result_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:  # Skip empty lines
                key, value = line.split(':')  # Split on the colon
                result_dict[key] = value
    return result_dict

def jaccard_similarity(x,y):
    intersection = len(set.intersection(*[set(x), set(y)]))
    union = len(set.union(*[set(x), set(y)]))
    return intersection/float(union)


# Define vector file to read
vector_filename = "vector_output.txt"

# Define a list of customer descriptions
customer_descriptions = [
    "Betty is a 32-year-old programmer from New York who is active in sports and technology.",
]

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # This is a smaller, faster model from Sentence Transformers
#model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Generate embeddings for each description
embeddings = model.encode(customer_descriptions)

# Convert embeddings to a dictionary format for easy access
customer_embeddings = {f"Customer_{i+1}": embeddings[i] for i in range(len(customer_descriptions))}

'''
# Display the embeddings
for customer, embedding in customer_embeddings.items():
    #print(f"{customer} embedding: {embedding[:5]}...")  # Display the first 5 values for brevity
    print(f"{customer} embedding: {embedding}")  # Display the first 5 values for brevity

# Write embbeddings to text files
vector_filename = "vector_output.txt"
with open(vector_filename, 'w') as f:
    for customer, embedding in customer_embeddings.items():
        f.write('{} {}\n'.format(customer, ' '.join(['{:e}'.format(item) for item in embedding])))
'''        

vector_dict = create_vector_dict(vector_filename)
#print(vector_dict)

print(type(customer_embeddings))
cust = customer_embeddings["Customer_1"] 
print(cust)

for key, value in vector_dict.items():
    # Need to read as np.float32 and not as str
    embbed = np.fromstring(value, sep=' ', dtype=np.float32)
    #print(type(embbed))
    #print(type(embbed[0]))
    #print(isinstance(embbed, np.ndarray))
    #print(isinstance(embbed[0], np.float64))
    #print(isinstance(embbed[0], np.double))
    #print(type(cust))
    #print(type(cust[0]))
    #print(isinstance(cust, np.ndarray))
    #print(isinstance(cust[0], np.float64))
    #print(isinstance(cust[0], np.double))
    # Compute cosine-similarits

    # Compute cosine similarities
    st_similarity = model.similarity(embbed, cust)
    print(f"Sentence Transformer similarity score for {key} : {st_similarity}")

    # Sentence Transformer cosine similarity
    st_cosine_scores = util.cos_sim(embbed, cust)
    print(f"Sentence Transformer Util cosine score for {key} : {st_cosine_scores}")

    # PyTorch cosine similarity
    pytorch_cosine_scores = util.pytorch_cos_sim(embbed, cust)
    print(f"Pytorch cosine score for {key} : {pytorch_cosine_scores}")
    

    # Jaccard similarity
    jaccard_score = jaccard_similarity(embbed, cust)
    print(f"Jaccard similarity score for {key} : {jaccard_score}")

    print("\n")

    '''
    #Output the pairs with their score
    for i in range(len(sentences1)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
    '''



