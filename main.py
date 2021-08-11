from utils.plotting import *
from utils.data_manipulation import *


bearings = pd.read_csv(r'data\bearings_final_master.csv')
axes_cumulative = pd.read_csv(r'data\bearing_grouped_axes_cumulative.csv')


if __name__=='__main__':

    only_failed_bearings = bearings[bearings.status == 0]
    columns_to_drop = ['status', 'bearing_id']
    only_failed_bearings.drop(columns_to_drop, axis=1, inplace=True)

    rename_dictionary_columns(axes_cumulative)
    axes_cumulative.set_index('experiment_id', inplace=True)
    axes_cumulative = axes_cumulative.iloc[:100, :]

    failed_bearings = pd.concat([only_failed_bearings, axes_cumulative], axis=1)
    failed_bearings = failed_bearings.dropna()

    failed_bearings_normal = (failed_bearings - failed_bearings.mean()) / failed_bearings.std()
    print(failed_bearings_normal.columns)

    ###########################################################
    # itertools to find best scores for two features combined #
    ###########################################################

    columns_list = failed_bearings_normal.columns
    clusters_amount = [2, 3, 4, 5]
    df_two_features = create_dataframe_selected_features_sorted_by_score(failed_bearings_normal, columns_list,
                                                                         clusters_amount, 2)
    df_two_features.sort_values(by=['silhouette_score'], ascending=False, inplace=True)
    df_two_features.to_csv('csv_output/scores_two_features.csv')

    two_feat_df = pd.read_csv('csv_output/scores_two_features.csv')

    plot_one_2d_with_silhouette_score(two_feat_df, failed_bearings_normal, 300)
    loop_over_all_2d_scatterplots(two_feat_df, failed_bearings_normal)

    # ################################################
    # choosing features based on itertools results: #
    # plot 2d every combination, starting from highest silhouette score to lowest
    # choose a plot that best shows clusters
    # decide on amount of clusters with elbow       #
    # ################################################

    df1 = failed_bearings_normal[['a1_y_mean', 'a2_x_mean']]
    n_clusters = [2, 3, 4, 5, 6]
    plot_elbow_kmeans(df1, n_clusters, 'kmeans model on features a1_y_mean and a2_x_mean')
    plot_scatter_cluster(df1, 3, 'clustering with k-means')

    df2 = failed_bearings_normal[['a2_x_cumul', 'a2_x_mean']]
    n_clusters = [2, 3, 4, 5, 6]
    plot_elbow_kmeans(df2, n_clusters, 'kmeans model on features a2_x_cumul and a2_x_mean')
    plot_scatter_cluster(df2, 3, 'clustering with k-means')

    # #############################################################
    # # itertools to find best scores for three features combined #
    # #############################################################
    df_three_features = create_dataframe_three_features(failed_bearings_normal,
                                                        columns_list, clusters_amount, 3)
    df_three_features.sort_values(by=['silhouette_score'], ascending=False, inplace=True)
    df_three_features.to_csv('csv_output/scores_three_features.csv')
    three_feat_df = pd.read_csv('csv_output/scores_three_features.csv')

    ################################################
    # choosing features based on itertools results #
    # with silhouette plotting                     #
    ################################################

    n_clusters = [2, 3, 4, 5, 6]
    for idx, row in three_feat_df.iterrows():
        f1 = row.feature_1
        f2 = row.feature_2
        f3 = row.feature_3
        df = failed_bearings_normal[[f1, f2, f3]]
        plot_silhouette_samples(df, n_clusters, idx)

    three_feat_df_cleaned = drop_rows_twoclusters_few_datapoints(three_feat_df)
    n_clusters = [2, 3, 4, 5, 6]

    # plot silhouette plots with samples
    row = three_feat_df_cleaned.loc[7214, :]
    f1 = row.feature_1
    f2 = row.feature_2
    f3 = row.feature_3
    cluster_n = row.cluster_number
    print(cluster_n)
    df = failed_bearings_normal[[f1, f2, f3]]
    plot_silhouette_samples(df, n_clusters, 7214)

    df3 = failed_bearings_normal[['a2_y_mean', 'a1_x_ff_range', 'a1_x_fft_max']]
    plot_elbow_kmeans(df3, n_clusters, 'elbow plot for three combined features: a2_y_mean, a1_x_ff_range, a1_x_fft_max')

    ##############################
    # try 6 features: unfinished #
    ##############################

    # columns = failed_bearings_normal.columns[:10]
    # sil_scores = []
    # cluster_numbers = []
    # labels_list = []
    # feature_1_all = []
    # feature_2_all = []
    # feature_3_all = []
    # feature_4_all = []
    # feature_5_all = []
    # feature_6_all = []
    #
    # n_clusters = [2, 3, 4]
    # combo_6 = list(itertools.combinations(columns, 6))
    # index = 0
    # for f1, f2, f3, f4, f5, f6 in combo_6:
    #     df_six = failed_bearings_normal[[f1, f2, f3, f4, f5, f6]]
    #     for cluster_nr in n_clusters:
    #         kmeans = KMeans(random_state=42, n_clusters=cluster_nr, init='k-means++')
    #         kmeans.fit(df_six)
    #         sil_score = silhouette_score(df_six, kmeans.labels_)
    #         sil_scores.append(sil_score)
    #         feature_1_all.append(f1)
    #         feature_2_all.append(f2)
    #         feature_3_all.append(f3)
    #         feature_4_all.append(f4)
    #         feature_5_all.append(f5)
    #         feature_6_all.append(f6)
    #         cluster_numbers.append(cluster_nr)
    #         labels_list.append(kmeans.labels_.tolist())
    #         print('index', index)
    #         index += 1
    #
    # df_six_features = pd.DataFrame({
    #     'silhouette_score': sil_scores,
    #     'feature_1': feature_1_all,
    #     'feature_2': feature_2_all,
    #     'feature_3': feature_3_all,
    #     'feature_4': feature_4_all,
    #     'feature_5': feature_5_all,
    #     'feature_6': feature_6_all,
    #     'cluster_number': cluster_numbers,
    #     'labels': labels_list
    # })

    # df_six_features.sort_values(by=['silhouette_score'], ascending=False, inplace=True)
    # df_six_features.to_csv('csv_output/scores_six_features.csv')

    # n_clusters = [2, 3, 4, 5, 6]
    # for idx, row in df_six_features.iterrows():
    #     f1 = row.feature_1
    #     f2 = row.feature_2
    #     f3 = row.feature_3
    #     f4 = row.feature_4
    #     f5 = row.feature_5
    #     f6 = row.feature_6
    #     df = failed_bearings_normal[[f1, f2, f3, f4, f5, f6]]
    #     plot_silhouette_samples(df, n_clusters, idx)

    # df6 = failed_bearings_normal[[]]    # ADD SIX FEATURES
    # plot_elbow_kmeans(df6, n_clusters, 'elbow plot for six combined features')

