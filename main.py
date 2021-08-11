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


    #####################################################
    # we can still look in 3d, but harder to interpret.   #
    # we still use elbow, but also introduce new metrics  #
    #####################################################

    row_number = 54000
    row = three_feat_df.iloc[row_number, :]
    plot_scatter_3d(failed_bearings_normal, row)

    ################################################
    # choosing features based on itertools results #
    # with silhouette plotting                     #
    ################################################

    # INTRODUCING SILHOUETTE PLOTTING
    n_clusters = [2, 3, 4, 5, 6]
    for idx, row in three_feat_df.iloc[300:, :].iterrows():
        f1 = row.feature_1
        f2 = row.feature_2
        f3 = row.feature_3
        df = failed_bearings_normal[[f1, f2, f3]]
        plot_silhouette_samples(df, n_clusters, idx)

    # DECIDE ON THREE FEATURES, THEN ELBOW PLOT IT
    # ELBOWING
    # plot_elbow_kmeans(df3, n_clusters, 'elbow plot')



    #############################################################################
    # REMINDER IF YOU LOOK FOR COMBO_: reduce the amount of features, clusters! #
    # 3 picks from 59 leads to 32.509 options, multiplied by n_clusters!        #
    # 3 picks from 50 leads to 19.600 options, multiplied by n_clusters!        #
    #############################################################################
