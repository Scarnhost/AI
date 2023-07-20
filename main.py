import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

def best_cluster(X):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    wcss_diff = np.diff(wcss)
    wcss_diff_percentage_change = (wcss_diff[:-1] - wcss_diff[1:]) / wcss_diff[:-1]
    elbow_index = np.argmax(wcss_diff_percentage_change < 0.1) + 1
    optimal_k = elbow_index+1
    best_wcss = wcss[elbow_index]
    print("Optimal number of clusters (k):", optimal_k)
    print("Best WCSS value:", best_wcss)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    return optimal_k






def k_clustering(window_size):
    dataset = pd.read_csv("Mall_Customers.csv")
    X=dataset.iloc[window_size:,[3,4]].values

    i=best_cluster(X)
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(X)
    #print(y_kmeans)
    #print(X)
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'pink']
    for cluster_num in range(i):
        plt.scatter(X[y_kmeans == cluster_num, 0], X[y_kmeans == cluster_num, 1], s=100, c=colors[cluster_num],
                    label=f'Cluster {cluster_num + 1}')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', label='Centroids')

    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    while True:
        user_input = input("Do you want to run programme(yes/no)? ").strip().lower()
        if user_input == "no":
            print("Exiting the programme")
            break
        elif user_input == "yes":
            window_size = int(input("Enter Current Window size: "))
            print("Window size = ",window_size)
            k_clustering(window_size)
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
