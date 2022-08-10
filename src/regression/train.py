import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

levels = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def parse_arguments():
    """Parse arguments and returns it.

    Args:
        None

    Returns:
        returns a args object with parse arguments

    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        type=str,
        help="Training dataset",
        default="data/processed/housing/train_data.csv",
    )
    parser.add_argument(
        "--train_labels",
        type=str,
        help="Training labels",
        default="data/processed/housing/train_label.csv",
    )
    parser.add_argument(
        "--pickle_output_folder",
        type=str,
        help="Output folder for pickle",
        default="artifacts",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        help="Logging file output",
        default="logs/train_log.txt",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        help="level of logging",
        default="debug",
        choices=[
            "critical",
            "error",
            "warn",
            "warning",
            "info",
            "debug",
        ],
    )
    return parser.parse_args()


def train():
    """Runs model on the training dataset.

    Args:
        None

    Returns:
        Returns the best model

    """

    args = parse_arguments()

    logging.basicConfig(
        filename=args.logfile, filemode="w", level=levels.get(args.loglevel.lower())
    )

    logging.info("Loading datasets")
    housing_prepared = pd.read_csv(args.train_data)
    housing_labels = pd.read_csv(args.train_labels)

    logging.info("Running linear regression")
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_rmse

    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    lin_mae

    logging.info("Running decision tree regression")
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_rmse

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    logging.info("Running random forest tree regression")
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    grid_search.best_params_
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    logging.info("Getting best model")
    final_model = grid_search.best_estimator_

    with open(os.path.join(args.pickle_output_folder, "best_model.pickle"), "wb") as f:
        pickle.dump(final_model, f)

    return final_model


if __name__ == "__main__":
    train()
