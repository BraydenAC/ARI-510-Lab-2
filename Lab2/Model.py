import sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import dbscan
from sklearn.metrics import silhouette_score
scaler = StandardScaler()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Loading dataset into project
X_train_dev = np.loadtxt('UCI HAR Dataset/train/X_train.txt')
y_train_dev = np.loadtxt('UCI HAR Dataset/train/y_train.txt')
X_test =      np.loadtxt('UCI HAR Dataset/test/X_test.txt')
y_test =      np.loadtxt('UCI HAR Dataset/test/y_test.txt')

#Split train into train and dev
X_train, X_dev, y_train, y_dev= train_test_split(X_train_dev, y_train_dev, test_size=0.1, random_state=42)

#Apply PCA
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#display hist
sns.histplot(y_train)
# plt.show()
plt.close()

#display line graph
explained_variance = pca.explained_variance_ratio_
components = np.arange(1, len(explained_variance) + 1)
sns.lineplot(x=components, y=explained_variance)
# plt.show()
plt.close()

#Scatter plot for first 2 principal components
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=100)

legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
plt.gca().add_artist(legend1)

plt.title("Scatter plot of the first 2 principal components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
# plt.show()
plt.close()

#Kmeans Clustering
KMeans_Model = KMeans(n_clusters=2, random_state=42)
KMeans_Model.fit(X_train)
print("Silhouette Score:")
print(silhouette_score(X_train, KMeans_Model.labels_))

#Scatter plot for cluster labels using top 2 principle components
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=KMeans_Model.labels_, cmap='viridis', edgecolor='k', s=100)

legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
plt.gca().add_artist(legend1)

plt.title("Scatter plot for cluster labels")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
# plt.show()
plt.close()

#DBSCAN
dbscan_model = DBSCAN(eps=5.03, min_samples=45, metric='euclidean')
dbscan_model.fit(X_train)

#Scatter plot for DBSCAN
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=dbscan_model.labels_, cmap='viridis', edgecolor='k', s=100)

legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
plt.gca().add_artist(legend1)

plt.title("Scatter plot for DBSCAN")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()
plt.close()