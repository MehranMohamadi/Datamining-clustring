import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Customers.csv')
X = dataset.iloc[:, [3, 4]].values



from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters(K)')
plt.ylabel('WCSS')
plt.show()



kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
Y_Kmeans = kmeans.fit_predict(X)

# Visualising the clusters

plt.scatter(X[Y_Kmeans == 0, 0], X[Y_Kmeans == 0, 1], s=100, c='red', label='Cluster 1')

plt.scatter(X[Y_Kmeans == 1, 0], X[Y_Kmeans == 1, 1], s=100, c='indigo', label='Cluster 2')

plt.scatter(X[Y_Kmeans == 2, 0], X[Y_Kmeans == 2, 1], s=100, c='magenta', label='Cluster 3')

plt.scatter(X[Y_Kmeans == 3, 0], X[Y_Kmeans == 3, 1], s=100, c='black', label='Cluster 4')

plt.scatter(X[Y_Kmeans == 4, 0], X[Y_Kmeans == 4, 1], s=100, c='lime', label='Cluster 5')

plt.scatter(X[Y_Kmeans == 5, 0], X[Y_Kmeans == 5, 1], s=100, c='green', label='Cluster 6')

plt.scatter(X[Y_Kmeans == 6, 0], X[Y_Kmeans == 6, 1], s=100, c='blue', label='Cluster 7')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')

plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()
