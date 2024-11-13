import numpy as np
from sklearn.cluster import KMeans

# Assuming 'embeddings' is a 2D numpy array containing your vector data
kmeans = KMeans(n_clusters=3)
kmeans.fit(embeddings)
labels = kmeans.labels_

# 'labels' now contains the cluster assignment for each data point