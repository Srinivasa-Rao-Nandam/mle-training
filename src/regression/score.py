import argparse
import logging
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

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


def score(
    test_data,
    test_labels,
    pickle_model,
    logfile="logs\\score_log.txt",
    loglevel="debug",
):
    """Returns score on test dataset.

    Args:
        None

    Returns:
        returns rmse and mse for housing dataset

    """

    logging.basicConfig(
        filename=logfile, filemode="w", level=levels.get(loglevel.lower())
    )

    mlflow.log_param(key="test_data_path", value=test_data)
    mlflow.log_param(key="test_data_path", value=test_labels)
    mlflow.log_param(key="model_path", value=pickle_model)

    logging.info("Loading datasets")
    X_test_prepared = pd.read_csv(test_data)
    y_test = pd.read_csv(test_labels)

    with open(pickle_model, "rb") as f:
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

    mlflow.log_metric(key="test_rmse", value=final_rmse)
    mlflow.log_metric(key="test_mse", value=final_mse)
    return final_rmse, final_mse


if __name__ == "__main__":
    args = parse_arguments()
    score(
        args.test_data, args.test_labels, args.pickle_model, args.logfile, args.loglevel
    )
