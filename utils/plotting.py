from ast import literal_eval

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


def plot_elbow_kmeans(df: pd.DataFrame, n_clusters: list, title: str) -> None:
    distances = []
    silo_scores = []
    for cluster_amount in n_clusters:
        kmeans = KMeans(random_state=42, n_clusters=cluster_amount).fit(df)
        distances.append(kmeans.inertia_)
        silo_scores.append(silhouette_score(df, kmeans.labels_))
    plt.plot(n_clusters, distances)
    plt.xlabel("Number of Clusters (k)")
    plt.xticks([2, 3, 4, 5, 6])
    plt.ylabel("Distance (inertia)")
    plt.title(title)
    plt.show()


def plot_scatter_cluster(df: pd.DataFrame, cluster_amount: int, title: str) -> None:
    kmeans = KMeans(random_state=42, n_clusters=cluster_amount).fit(df)
    df['cluster'] = kmeans.labels_
    sil_score = silhouette_score(df, kmeans.labels_)
    sns.scatterplot(x=df.iloc[:,0], y=df.iloc[:,1], hue=df.cluster, palette='Set2')
    plt.title((title, str(sil_score)))
    plt.show()


def plot_scatter_3d(df: pd.DataFrame, row: pd.Series) -> None:
    cluster = literal_eval(row[5])
    sizes = [5] * len(cluster)
    fig = px.scatter_3d(df, x=row[1], y=row[2], z=row[3], size=sizes, size_max=5,
                         color=cluster, width=800, height=800)
    fig.update_layout(title="clustering")
    fig.show()


def plot_silhouette_samples(df: pd.DataFrame, n_clusters: list, row_index: int) -> None:
    silhouette_scores_n_clusters = []
    for n_cluster in n_clusters:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(df) + (n_cluster+1) * 10])

        kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        cluster_labels = kmeans.fit_predict(df)
        silhouette_score_df = silhouette_score(df, cluster_labels)

        print("For n_clusters =", n_cluster, "The average silhouette_score is :", silhouette_score_df)
        silhouette_scores_n_clusters.append(silhouette_score_df)
        sample_silhouette_values = silhouette_samples(df, cluster_labels)

        y_lower = 10
        for idx in range(n_cluster):
            in_cluster_silhouette_values = sample_silhouette_values[kmeans.labels_==idx]
            in_cluster_silhouette_values.sort()
            size_cluster_idx = in_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_idx
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, in_cluster_silhouette_values,
                              alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_cluster_idx, str(idx))
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_score_df, color="red", linestyle="--")
        ax.set_yticks([])
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        title = f"row {row_index} " \
                f"with n_clusters = {n_cluster}"
        plt.suptitle(title,
                     fontsize=14, fontweight='bold')
        plt.show()


def loop_over_all_2d_scatterplots(df: pd.DataFrame, original_df: pd.DataFrame) -> None:
    for idx, row in df.iterrows():
        f1 = row.feature_1
        f2 = row.feature_2
        # labels = literal_eval(row.labels)
        labels = row.labels
        sil_sco = row.silhouette_score
        sns.scatterplot(x=original_df[f1],
                        y=original_df[f2],
                         hue=labels,
                         palette='Set2')
        plt.title('silhouette score: ' + str(round(sil_sco, 2)))
        plt.show()


def plot_one_2d_with_silhouette_score(df: pd.DataFrame, original_df, row: int) -> None:
    f1 = df.iloc[row].feature_1
    f2 = df.iloc[row].feature_2
#   labels = literal_eval(df.iloc[row].labels)
    labels = df.iloc[row].labels
    sil_sco = df.iloc[row].silhouette_score
    sns.scatterplot(x=original_df[f1],
                    y=original_df[f2],
                    hue=labels,
                    palette='Set2')
    plt.title('silhouette score: ' + str(round((sil_sco), 2)))
    plt.show()


def plot_one_2d_with_silhouette_score_based_on_features(feature_1, feature_2,
                                                        df: pd.DataFrame, original_df) -> None:
    df_to_plot = df[(df.feature_1==feature_1) & (df.feature_2==feature_2)]
#   labels = literal_eval(df_to_plot.labels)
    labels = df_to_plot.labels
    sil_sco = df_to_plot.silhouette_score
    sns.scatterplot(x=original_df[feature_1],
                    y=original_df[feature_2],
                    hue=labels,
                    palette='Set2')
    plt.title('silhouette score: ' + str(round((sil_sco), 2)))
    plt.show()
