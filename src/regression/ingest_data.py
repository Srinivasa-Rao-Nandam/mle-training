import argparse
import logging
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

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


def transform_data():
    """Transform data and stores it

    Args:
        None

    Returns:
        returns None, stores files at particular location

    """

    args = parse_arguments()
    logging.basicConfig(
        filename=args.logfile, filemode="w", level=levels.get(args.loglevel.lower())
    )

    HOUSING_PATH = os.path.join(args.raw_dataset_folder, "housing")
    HOUSING_URL = args.download_url
    OUTPUT_HOUSING_PATH = os.path.join(args.processed_dataset_folder, "housing")
    os.makedirs(OUTPUT_HOUSING_PATH, exist_ok=True)

    logging.info("downloading dataset")
    fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    housing = load_housing_data(HOUSING_PATH)

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
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    OUTPUT_HOUSING_TRAINING_DATA_PATH = os.path.join(
        OUTPUT_HOUSING_PATH, "train_data.csv"
    )
    OUTPUT_HOUSING_TRAINING_LABEL_PATH = os.path.join(
        OUTPUT_HOUSING_PATH, "train_label.csv"
    )
    logging.info("saving train dataset")
    housing_prepared.to_csv(OUTPUT_HOUSING_TRAINING_DATA_PATH)
    housing_labels.to_csv(OUTPUT_HOUSING_TRAINING_LABEL_PATH)

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    OUTPUT_HOUSING_TEST_DATA_PATH = os.path.join(OUTPUT_HOUSING_PATH, "test_data.csv")
    OUTPUT_HOUSING_TEST_LABEL_PATH = os.path.join(OUTPUT_HOUSING_PATH, "test_label.csv")

    logging.info("saving test dataset")
    X_test_prepared.to_csv(OUTPUT_HOUSING_TEST_DATA_PATH)
    y_test.to_csv(OUTPUT_HOUSING_TEST_LABEL_PATH)


if __name__ == "__main__":
    transform_data()
