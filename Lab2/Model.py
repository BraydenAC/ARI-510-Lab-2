from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
scaler = StandardScaler()

import seaborn as sns
import matplotlib.pyplot as plt
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
plt.show()
plt.close()

#display line graph
explained_variance = pca.explained_variance_ratio_
components = np.arange(1, len(explained_variance) + 1)
sns.lineplot(x=components, y=explained_variance)
plt.show()
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
plt.show()
plt.close()

#Kmeans Clustering
KMeans_Model = KMeans(n_clusters=2, random_state=42)
KMeans_Model.fit(X_train)
print("Silhouette Score:")
print(silhouette_score(X_train, KMeans_Model.labels_))

#Scatter plot for cluster labels using top 2 principal components
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=KMeans_Model.labels_, cmap='viridis', edgecolor='k', s=100)

legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
plt.gca().add_artist(legend1)

plt.title("Scatter plot for cluster labels")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()
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

X_train_first = X_train_pca[:, :2]
X_test_first = X_test_pca[:, :2]
X_train_top = np.delete(X_train_pca[:, :3], 1, axis=1)
X_test_top = np.delete(X_test_pca[:, :3], 1, axis=1)

#Random Forest Model: top 2 features
RandTree_Model = RandomForestClassifier()
RandTree_Model.fit(X_train_pca, y_train)
#Feature Importances
print()
print(RandTree_Model.feature_importances_)

#LOGISISTIC REGRESSION
#Top 2 features(first and third)
LogReg_Model1 = LogisticRegression(max_iter=500)
LogReg_Model1.fit(X_train_top, y_train)

#full feature set
LogReg_Model2 = LogisticRegression(max_iter=500)
LogReg_Model2.fit(X_train, y_train)

#first two features
LogReg_Model3 = LogisticRegression(max_iter=500)
LogReg_Model3.fit(X_train_first, y_train)


#Make Predictions
Model_1_Predictions = LogReg_Model1.predict(X_test_top)
Model_2_Predictions = LogReg_Model2.predict(X_test)
Model_3_Predictions = LogReg_Model3.predict(X_test_first)

# Display Results
print(f"Logistic Regression, top 2: {accuracy_score(y_test, Model_1_Predictions)}")
print(f"Logistic Regression, Full: {accuracy_score(y_test, Model_2_Predictions)}")
print(f"Logistic Regression, first 2: {accuracy_score(y_test, Model_3_Predictions)}")
print()
print("Logistic Regression, top 2")
print(classification_report(y_test, Model_1_Predictions))
print("\nLogistic Regression, Full")
print(classification_report(y_test, Model_2_Predictions))
print("\nLogistic Regression, first 2")
print(classification_report(y_test, Model_3_Predictions))