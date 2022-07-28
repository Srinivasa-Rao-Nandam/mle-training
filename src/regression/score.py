import argparse
import logging
import os
import pickle
import tarfile

import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
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
        "--test_data",
        type=str,
        help="Training dataset",
        default="data\\processed\\housing\\test_data.csv",
    )
    parser.add_argument(
        "--test_labels",
        type=str,
        help="Training labels",
        default="data\\processed\\housing\\test_label.csv",
    )
    parser.add_argument(
        "--pickle_model",
        type=str,
        help="Output folder for pickle",
        default="artifacts\\best_model.pickle",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        help="Logging file output",
        default="logs\\score_log.txt",
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


def score():
    """Returns score on test dataset.

    Args:
        None

    Returns:
        returns rmse and mse for housing dataset

    """

    args = parse_arguments()

    logging.basicConfig(
        filename=args.logfile, filemode="w", level=levels.get(args.loglevel.lower())
    )

    logging.info("Loading datasets")
    X_test_prepared = pd.read_csv(args.test_data)
    y_test = pd.read_csv(args.test_labels)

    with open(args.pickle_model, "rb") as f:
        final_model = pickle.load(f)

    logging.info("Running predictions")
    final_predictions = final_model.predict(X_test_prepared)

    logging.info("Calculating metrics")
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    print(
        "Score of best model, Final RMSE: {}, Final MSE: {}".format(
            final_rmse, final_mse
        )
    )
    return final_rmse, final_mse


if __name__ == "__main__":
    score()
