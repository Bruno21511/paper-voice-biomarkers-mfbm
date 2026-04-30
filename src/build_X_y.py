# -*- coding: utf-8 -*-
import numpy as np


def build_X_y(df_subset, label_map, mean_band, std_band):
    """
    Returns:
        X: features (n_samples, 2)
        y: labels
    """

    X = []
    y = []

    for _, row in df_subset.iterrows():

        mean_vec = np.array(row['mean_MFBM'])
        std_vec  = np.array(row['std_MFBM'])

        features = [
            mean_vec[mean_band],
            std_vec[std_band]
        ]

        X.append(features)
        y.append(label_map[row['group']])

    return np.array(X), np.array(y)