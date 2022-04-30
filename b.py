def knn_train_test4(_train_cols4, target_col, df4):
    """roda o knn test usando valor de k padrao e retorna o RMSE"""
    np.random.seed(1)

    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df4.index)
    rand_df = df4.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)

    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]

    k_values = list(range(1, 25))
    k_rmses = {}

    for k_value_4 in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k_value_4)
        knn.fit(train_df[_train_cols4], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[_train_cols4])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)

        k_rmses[k_value_4] = rmse
    return k_rmses


k_rmse_results = {}

for nr_best_feats in range(2, 6):
    k_rmse_results[f'{nr_best_feats} best features'] = knn_train_test4(
        sorted_features[:nr_best_feats],
        'price',
        numeric_cars
    )

print(k_rmse_results)
