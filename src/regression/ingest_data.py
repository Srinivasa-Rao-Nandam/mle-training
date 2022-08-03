import argparse
import logging
import os
import tarfile

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

levels = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def parse_arguments():
    """Parse arguments and returns it.

    Args:
        None

    Returns:
        returns a args object with parse arguments

    """

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--download_url",
        type=str,
        help="Download url from which to download the dataset",
        default="https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz",
    )
    argparser.add_argument(
        "--raw_dataset_folder",
        type=str,
        help="Folder to store raw dataset",
        default="data\\raw",
    )
    argparser.add_argument(
        "--processed_dataset_folder",
        type=str,
        help="Folder to store processed dataset",
        default="data\\processed",
    )
    argparser.add_argument(
        "--logfile",
        type=str,
        help="Logging file output",
        default="logs\\data_log.txt",
    )
    argparser.add_argument(
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
    return argparser.parse_args()


def fetch_housing_data(housing_url, housing_path):
    """Returns score on test dataset.

    Args:
        housing_url: Path to download housing url
        housing_path: Path to store housing dataset

    Returns:
        returns None value

    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    """Returns score on test dataset.

    Args:
        housing_path: Path where housing dataset is present

    Returns:
        returns a pandas object which has housing data

    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    """Returns score on test dataset.

    Args:
        data: Pandas object which has housing object

    Returns:
        returns a pandas object income category count

    """
    return data["income_cat"].value_counts() / len(data)


def transform_data(
    download_url,
    raw_dataset_folder,
    processed_dataset_folder,
    logfile="logs\\data_log.txt",
    loglevel="debug",
):
    """Transform data and stores it

    Args:
        None

    Returns:
        returns None, stores files at particular location

    """
    logging.basicConfig(
        filename=logfile, filemode="w", level=levels.get(loglevel.lower())
    )
    mlflow.log_param(key="download_url", value=download_url)

    HOUSING_PATH = os.path.join(raw_dataset_folder, "housing")
    HOUSING_URL = download_url
    OUTPUT_HOUSING_PATH = os.path.join(processed_dataset_folder, "housing")
    os.makedirs(OUTPUT_HOUSING_PATH, exist_ok=True)

    logging.info("downloading dataset")
    fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    housing = load_housing_data(HOUSING_PATH)

    mlflow.log_param(key="raw_dataset_folder", value=raw_dataset_folder)

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    logging.info("splitting the dataset")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ]
    )

    housing_prepared = full_pipeline.fit_transform(housing)

    OUTPUT_HOUSING_TRAINING_DATA_PATH = os.path.join(
        OUTPUT_HOUSING_PATH, "train_data.csv"
    )
    OUTPUT_HOUSING_TRAINING_LABEL_PATH = os.path.join(
        OUTPUT_HOUSING_PATH, "train_label.csv"
    )
    logging.info("saving train dataset")
    pd.DataFrame(housing_prepared).to_csv(OUTPUT_HOUSING_TRAINING_DATA_PATH)
    pd.DataFrame(housing_labels).to_csv(OUTPUT_HOUSING_TRAINING_LABEL_PATH)

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)

    num_attribs = list(X_test_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ]
    )

    X_test_prepared = full_pipeline.fit_transform(X_test)

    OUTPUT_HOUSING_TEST_DATA_PATH = os.path.join(OUTPUT_HOUSING_PATH, "test_data.csv")
    OUTPUT_HOUSING_TEST_LABEL_PATH = os.path.join(OUTPUT_HOUSING_PATH, "test_label.csv")

    logging.info("saving test dataset")
    pd.DataFrame(X_test_prepared).to_csv(OUTPUT_HOUSING_TEST_DATA_PATH)
    pd.DataFrame(y_test).to_csv(OUTPUT_HOUSING_TEST_LABEL_PATH)
    mlflow.log_param(key="processed_dataset_folder", value=processed_dataset_folder)

    return (
        OUTPUT_HOUSING_TRAINING_DATA_PATH,
        OUTPUT_HOUSING_TRAINING_LABEL_PATH,
        OUTPUT_HOUSING_TEST_DATA_PATH,
        OUTPUT_HOUSING_TEST_LABEL_PATH,
    )


if __name__ == "__main__":
    args = parse_arguments()
    transform_data(
        args.download_url,
        args.raw_dataset_folder,
        args.processed_dataset_folder,
        args.logfile,
        args.loglevel,
    )
