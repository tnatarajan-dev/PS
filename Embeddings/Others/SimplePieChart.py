import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate or load your embedded vector
# For this example, let's create a random vector with 1024 dimensions
vector = np.random.rand(1024)

# Step 2: Segment the vector
# Define the number of segments, e.g., 8 segments
num_segments = 8
segment_size = len(vector) // num_segments
segments = [vector[i * segment_size:(i + 1) * segment_size] for i in range(num_segments)]

# Step 3: Calculate magnitude or sum for each segment
# We’ll use the sum here for simplicity
segment_sums = [np.sum(segment) for segment in segments]

# Step 4: Plot a pie chart
labels = [f'Segment {i+1}' for i in range(num_segments)]
plt.figure(figsize=(8, 8))
plt.pie(segment_sums, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Segmented Embedding Vector')
plt.show()

