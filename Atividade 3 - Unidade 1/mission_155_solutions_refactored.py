"""
Author: Yolanda Dantas
Date: apr. 2022.
Exercise 3, unit 1
Goal: refactor a machine learning code
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

pd.options.display.max_columns = 99

cols = ['symboling', 'normalized-losses', 'make', 'fuel-type',
        'aspiration', 'num-of-doors', 'body-style', 'drive-wheels',
        'engine-location', 'wheel-base', 'length', 'width', 'height',
        'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size',
        'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower',
        'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv('imports-85.data', names=cols)

cars.head()

# Select only the columns with continuous values from -
# https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names
continuous_values_cols = ['normalized-losses', 'wheel-base', 'length',
                          'width', 'height', 'curb-weight', 'engine-size',
                          'bore', 'stroke', 'compression-rate', 'horsepower',
                          'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
numeric_cars = cars[continuous_values_cols]

numeric_cars.head(5)

numeric_cars = numeric_cars.replace('?', np.nan)
numeric_cars.head(5)

numeric_cars = numeric_cars.astype('float')
numeric_cars.isnull().sum()

# Because `price` is the column we want to predict,
# let's remove any rows with missing `price` values.
numeric_cars = numeric_cars.dropna(subset=['price'])
numeric_cars.isnull().sum()

# Replace missing values in other columns using column means.
numeric_cars = numeric_cars.fillna(numeric_cars.mean())

# Confirm that there's no more missing values!
numeric_cars.isnull().sum()

# Normalize all columnns to range from 0 to 1 except the target column.
price_col = numeric_cars['price']
numeric_cars = (numeric_cars - numeric_cars.min()) / \
    (numeric_cars.max() - numeric_cars.min())
numeric_cars['price'] = price_col


def knn_train_test_1(train_col, target_col, _df):
    """Run KNN test using the default k value and return the RMSE"""
    knn = KNeighborsRegressor()
    np.random.seed(1)

    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(_df.index)
    rand_df = _df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)

    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]

    # Fit a KNN model using default k value.
    knn.fit(train_df[[train_col]], train_df[target_col])

    # Make predictions using model.
    predicted_labels = knn.predict(test_df[[train_col]])

    # Calculate and return RMSE.
    mse = mean_squared_error(test_df[target_col], predicted_labels)
    rmse = np.sqrt(mse)
    return rmse


rmse_results = {}
train_cols = numeric_cars.columns.drop('price')

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `rmse_results`.
for col in train_cols:
    rmse_val = knn_train_test_1(col, 'price', numeric_cars)
    rmse_results[col] = rmse_val

# Create a Series object from the dictionary so
# we can easily view the results, sort, etc
rmse_results_series = pd.Series(rmse_results)
rmse_results_series.sort_values()


def knn_train_test_2(train_col, target_col, _df):
    """Run KNN test using different k values and return the RMSE"""
    np.random.seed(1)

    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(_df.index)
    rand_df = _df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)

    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]

    k_values = [1, 3, 5, 7, 9]
    k_rmses = {}

    for k_value in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k_value)
        knn.fit(train_df[[train_col]], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[[train_col]])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)

        k_rmses[k_value] = rmse
    return k_rmses


k_rmse_results = {}

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `rmse_results`.
train_cols = numeric_cars.columns.drop('price')
for col in train_cols:
    rmse_val = knn_train_test_2(col, 'price', numeric_cars)
    k_rmse_results[col] = rmse_val

print(k_rmse_results)

# %matplotlib inline

for k, v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())

    plt.plot(x, y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')

    # Compute average RMSE across different `k` values for each feature.
feature_avg_rmse = {}
for k, v in k_rmse_results.items():
    avg_rmse = np.mean(list(v.values()))
    feature_avg_rmse[k] = avg_rmse
series_avg_rmse = pd.Series(feature_avg_rmse)
sorted_series_avg_rmse = series_avg_rmse.sort_values()
print(sorted_series_avg_rmse)

sorted_features = sorted_series_avg_rmse.index


def knn_train_test_3(train_cols_3, target_col, _df):
    """Run KNN test using 5 as k value and return the RMSE"""
    np.random.seed(1)

    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(_df.index)
    rand_df = _df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)

    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]

    k_values = [5]
    k_rmses = {}

    for k_value in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k_value)
        knn.fit(train_df[train_cols_3], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[train_cols_3])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)

        k_rmses[k_value] = rmse
    return k_rmses


k_rmse_results = {}

for nr_best_feats in range(2, 7):
    k_rmse_results[f'{nr_best_feats} best features'] = knn_train_test_3(
        sorted_features[:nr_best_feats],
        'price',
        numeric_cars
    )

print(k_rmse_results)

def knn_train_test_4(train_cols_4, target_col, _df):
    """Run KNN test using different k values and return the RMSE"""
    np.random.seed(1)
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(_df.index)
    rand_df = _df.reindex(shuffled_index)
    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]

    k_values = list(range(1, 25))
    k_rmses = {}

    for k_value in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k_value)
        knn.fit(train_df[train_cols_4], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[train_cols_4])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)

        k_rmses[k_value] = rmse
    return k_rmses


k_rmse_results = {}

for nr_best_feats in range(2, 6):
    k_rmse_results[f'{nr_best_feats} best features'] = knn_train_test_4(
        sorted_features[:nr_best_feats],
        'price',
        numeric_cars
    )

print(k_rmse_results)

for k, v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    plt.plot(x, y, label="{k}")

plt.xlabel('k value')
plt.ylabel('RMSE')
plt.legend()
