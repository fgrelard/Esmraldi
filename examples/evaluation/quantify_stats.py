"""
Quantify statistical analysis
and clustering
through silhouette profile and scores
"""
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_silhouette(silhouette_values, labels):
    """
    Plot the silhouette coefficient

    Parameters
    ----------
    silhouette_values: list
        silhouette values
    labels: list
        list of labels
    """
    fig, ax1 = plt.subplots(1, 1)
    n_clusters = labels.max() + 1
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    #ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    y_lower = 10
    for i in range(n_clusters):
         # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.35, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label", labelpad=20)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=np.mean(silhouette_values), color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()

labels = np.load("data/labels_maldi.npy")
image_maldi = np.load("data/after_registration_maldi.npy")

silhouette_values = metrics.silhouette_samples(image_maldi, labels)
silhouette_score = metrics.silhouette_score(image_maldi, labels)
neg = silhouette_values[silhouette_values < 0]
print("Size negative silhouette score=", len(neg), " mean=", np.mean(neg), " stddev=", np.std(neg))
plot_silhouette(silhouette_values, labels)
print(np.mean(silhouette_values))
