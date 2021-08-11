from ast import literal_eval

import pandas as pd
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def rename_dictionary_columns(df: pd.DataFrame) -> None:
    rename_dict = {
        'a1_x': 'a1_x_cumul',
        'a1_y': 'a1_y_cumul',
        'a1_z': 'a1_z_cumul',
        'a2_x': 'a2_x_cumul',
        'a2_y': 'a2_y_cumul',
        'a2_z': 'a2_z_cumul',
    }
    df.rename(columns=rename_dict, inplace=True)


def create_dataframe_selected_features_sorted_by_score(df: pd.DataFrame, columns: list,
                                                       n_clusters: list, n_features: int) -> pd.DataFrame:
    sil_scores = []
    cluster_numbers = []
    labels_list = []
    feature_1_all = []
    feature_2_all = []

    combo_2 = list(itertools.combinations(columns, n_features))
    index = 0
    for f1, f2 in combo_2:
        df_two = df[[f1, f2]]
        for cluster_nr in n_clusters:
            kmeans = KMeans(random_state=42, n_clusters=cluster_nr, init='k-means++')
            kmeans.fit(df_two)
            sil_score = silhouette_score(df_two, kmeans.labels_)
            sil_scores.append(sil_score)
            feature_1_all.append(f1)
            feature_2_all.append(f2)
            cluster_numbers.append(cluster_nr)
            labels_list.append(kmeans.labels_.tolist())
            print('index', index)
            index += 1

    df_two_features = pd.DataFrame({
        'silhouette_score': sil_scores,
        'feature_1': feature_1_all,
        'feature_2': feature_2_all,
        'cluster_number': cluster_numbers,
        'labels': labels_list
    })

    return df_two_features


def create_dataframe_three_features(df: pd.DataFrame, columns: list,
                                    n_clusters: list, n_features: int) -> pd.DataFrame:
    sil_scores = []
    feature_1_all = []
    feature_2_all = []
    feature_3_all = []
    cluster_numbers = []
    labels_list = []

    combo_3 = list(itertools.combinations(columns, 3))
    index = 0
    for f1, f2, f3 in combo_3:
        df_three = df[[f1, f2, f3]]
        for cluster_nr in n_clusters:
            kmeans = KMeans(random_state=42, n_clusters=cluster_nr, init='k-means++')
            kmeans.fit(df_three)
            sil_score = silhouette_score(df_three, kmeans.labels_)
            sil_scores.append(sil_score)
            feature_1_all.append(f1)
            feature_2_all.append(f2)
            feature_3_all.append(f3)
            cluster_numbers.append(cluster_nr)
            labels_list.append(kmeans.labels_.tolist())
            print('index', index)
            index += 1

    df_three_features = pd.DataFrame({
        'silhouette_score': sil_scores,
        'feature_1': feature_1_all,
        'feature_2': feature_2_all,
        'feature_3': feature_3_all,
        'cluster_number': cluster_numbers,
        'labels': labels_list
    })

    return df_three_features


def drop_rows_twoclusters_few_datapoints(df: pd.DataFrame):
    df_to_return = df.copy()
    threshold = 10
    for idx, row in df_to_return.iterrows():
        if row.cluster_number == 2:
            labels = literal_eval(row.labels)
            labels = pd.Series(labels)
            vc = labels.value_counts()
            for value in vc.values:
                if value < threshold:
                    df_to_return.drop(df_to_return.index[idx], inplace=True)

    return df_to_return