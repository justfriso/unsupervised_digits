import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

digits = load_digits()
data = digits.data
labels_true = digits.target

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(data)
labels_pred = kmeans.labels_

mapping = {}
for i in range(num_clusters):
    labels = labels_true[labels_pred == i]
    counts = np.bincount(labels)
    mapping[i] = np.argmax(counts)

predicted_labels = [mapping[label] for label in labels_pred]

accuracy = accuracy_score(labels_true, predicted_labels)
print("Accuracy: ", accuracy)

plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    plt.scatter(data_pca[labels_pred == i, 0], data_pca[labels_pred == i, 1], label=str(mapping[i]))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, color='red', label='Centers')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('Handwritten Digit Clustering')
plt.show()
