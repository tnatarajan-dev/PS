import numpy as np

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_v1 = np.linalg.norm(vector1)
    norm_v2 = np.linalg.norm(vector2)
    return dot_product / (norm_v1 * norm_v2)

# Example usage
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

similarity = cosine_similarity(vec1, vec2)
print(similarity)
