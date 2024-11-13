import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# using sklearn
A = [1, 2, 3]
B = [4, 5, 6]

# reshape vectors as 2d arrays for cosine_similarity
A1 = np.array(A).reshape(1, -1)
B1 = np.array(B).reshape(1, -1)

similarity = cosine_similarity(A1, B1)
print(f"Cosine Similarity using Sklearn: {similarity}")


# using numpy
dot_product = np.dot(A, B)
magnitude_A = np.linalg.norm(A)
magnitude_B = np.linalg.norm(B)

cosine_similarity = dot_product / (magnitude_A * magnitude_B)
print(f"Cosine Similarity using NumPy: {cosine_similarity}")