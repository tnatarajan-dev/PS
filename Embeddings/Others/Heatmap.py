import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (dictionary of embeddings) for two customers
data = {
    "Customer_1": [-3.277246e-02, -3.013667e-02, 5.197407e-02, 9.218981e-03, -3.132878e-02, 2.771347e-03, 1.098836e-02, 1.333985e-02],
    "Customer_2": [2.857493e-02, -8.155182e-02, 7.172093e-02, 1.018526e-03, -6.902047e-02, 5.190836e-02, 3.302243e-02, 8.522870e-03],
    "Customer_3": [4.581115e-02, -1.659911e-02, 7.286185e-02, 3.629786e-02, -5.894421e-02, 4.431242e-02, -2.481864e-02, 4.034542e-02]
}

# Convert the dictionary to a 2D numpy array
embeddings = np.array(list(data.values()))

# Generate the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(embeddings, cmap='viridis', xticklabels=False, yticklabels=data.keys())
plt.title("Embedding Heatmap for Customers")
plt.xlabel("Embedding Dimensions")
plt.ylabel("Customers")
plt.show()
