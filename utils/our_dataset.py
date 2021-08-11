import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.cm as cm
import matplotlib.style as style

our_data = pd.read_csv(r'../data/bearing_final_df.csv')
our_data_failed = our_data[our_data.target == 0]
print(our_data_failed.columns)
unwanted_columns = ['experiment_id', 'Unnamed: 0', 'target']
our_data_failed_dropped = our_data_failed.drop(unwanted_columns, axis=1)
print(our_data_failed_dropped)

# pairplot to find the best point of departure
# sns.pairplot(our_data_failed_dropped)
# plt.show()

# 1: best two features to start (clear separation on plot)
x_meanvibr_maxampl = our_data_failed_dropped[['mean_vibrations_x', 'max_amplitudes_x']]
# normalize
x_meanvibr_maxampl_norm = (x_meanvibr_maxampl - x_meanvibr_maxampl.mean()) / x_meanvibr_maxampl.std()
for clusters_amount in np.arange(2, 5):
    print('kmeans clustering: cluster amount', clusters_amount)
    kmeans = KMeans(n_clusters=clusters_amount, init='k-means++', random_state=42).fit(x_meanvibr_maxampl_norm)
    print('silhouette_score:', silhouette_score(x_meanvibr_maxampl_norm, kmeans.labels_))
    x_meanvibr_maxampl_norm['cluster'] = kmeans.labels_
    print(x_meanvibr_maxampl_norm.cluster.value_counts())
    sns.scatterplot(x='mean_vibrations_x', y='max_amplitudes_x', hue='cluster', data=x_meanvibr_maxampl_norm)
    plt.show()

range_n_clusters = [2, 3, 4, 5, 6]
silhouette_avg_n_clusters = []

for n_clusters in range_n_clusters:
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(three_features_norm) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    cluster_labels = clusterer.fit_predict(three_features_norm)

    silhouette_avg = silhouette_score(three_features_norm, cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    silhouette_avg_n_clusters.append(silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(three_features_norm, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        in_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        in_cluster_silhouette_values.sort()
        size_cluster_i = in_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, in_cluster_silhouette_values,
                          alpha=0.7)
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
              "with n_clusters = %d" % n_clusters),
             fontsize=14, fontweight='bold')

plt.show()

style.use("fivethirtyeight")
plt.plot(range_n_clusters, silhouette_avg_n_clusters)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("silhouette score")
plt.show()

##### copy paste elbow method to determine ideal amount of clusters: 4
range_n_clusters = [1, 2, 3, 4, 5, 6]
avg_distance = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42).fit(x_meanvibr_maxampl_norm)
    avg_distance.append(clusterer.inertia_)

style.use("fivethirtyeight")
plt.plot(range_n_clusters, avg_distance)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Distance")
plt.show()

# TWO FEATURES:
# sil sco = 0.6076073668038054 for best 4 clusters


### add one feature
three_features = our_data_failed_dropped[['mean_vibrations_x',
                                          'max_amplitudes_x',
                                          'mean_vibrations_z']]
three_features_norm = (three_features - three_features.mean()) / three_features.std()
for clusters_amount in np.arange(2, 7):
    print('kmeans clustering: cluster amount', clusters_amount)
    kmeans = KMeans(n_clusters=clusters_amount, init='k-means++', random_state=42).fit(three_features_norm)
    print('silhouette_score:', silhouette_score(three_features_norm, kmeans.labels_))
    three_features_norm['cluster'] = kmeans.labels_
    print(three_features_norm.cluster.value_counts())

#### copy paste silhouette plot
range_n_clusters = [2, 3, 4, 5, 6]
silhouette_avg_n_clusters = []

for n_clusters in range_n_clusters:
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(three_features_norm) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    cluster_labels = clusterer.fit_predict(three_features_norm)

    silhouette_avg = silhouette_score(three_features_norm, cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    silhouette_avg_n_clusters.append(silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(three_features_norm, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        in_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        in_cluster_silhouette_values.sort()
        size_cluster_i = in_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, in_cluster_silhouette_values,
                          alpha=0.7)
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    # colors = plt.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    # ax2.scatter(x=three_features_norm.mean_vibrations_x, y=three_features_norm.max_amplitudes_x,
    #             marker='.', s=30, lw=0, alpha=0.7,
    #             c=colors,
    # edgecolor='k')

    # Labeling the clusters
    # centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
    #             c="white", alpha=1, s=200, edgecolor='k')

    # for i, c in enumerate(centers):
    #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
    #                 s=50, edgecolor='k')

    # ax2.set_title("The visualization of the clustered data.")
    # ax2.set_xlabel("Feature space for the 1st feature")
    # ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

style.use("fivethirtyeight")
plt.plot(range_n_clusters, silhouette_avg_n_clusters)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("silhouette score")
plt.show()

# THREE FEATURES RESULT:
# For n_clusters = 4 The average silhouette_score is : 0.6665314958009845           +.06

# 4: + max_amplitudes_z
# four_features = our_data_failed_dropped[['mean_vibrations_x',
#                                          'max_amplitudes_x',
#                                          'mean_vibrations_z',
#                                          'max_amplitudes_z'
#                                          ]]
# four_features_norm = (four_features - four_features.mean()) / four_features.std()
# for clusters_amount in np.arange(2, 7):
#     print('kmeans clustering: cluster amount', clusters_amount)
#     kmeans = KMeans(n_clusters=clusters_amount, init='k-means++', random_state=42).fit(four_features_norm)
#     print('silhouette_score:', silhouette_score(four_features_norm, kmeans.labels_))
#     four_features_norm['cluster'] = kmeans.labels_
#     print(four_features_norm.cluster.value_counts())


# RESULTS: NOT HAPPY: losing points, so add another 4the

# 4: mean_vibrations_y
four_features = our_data_failed_dropped[['mean_vibrations_x',
                                         'max_amplitudes_x',
                                         'mean_vibrations_z',
                                         'max_amplitudes_y'
                                         ]]
four_features_norm = (four_features - four_features.mean()) / four_features.std()
for clusters_amount in np.arange(2, 7):
    print('kmeans clustering: cluster amount', clusters_amount)
    kmeans = KMeans(n_clusters=clusters_amount, init='k-means++', random_state=42).fit(four_features_norm)
    print('silhouette_score:', silhouette_score(four_features_norm, kmeans.labels_))
    four_features_norm['cluster'] = kmeans.labels_
    print(four_features_norm.cluster.value_counts())
