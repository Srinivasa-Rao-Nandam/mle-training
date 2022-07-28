import os

from src.regression import ingest_data


def test_args():
    args = ingest_data.parse_arguments()
    assert (
        args.download_url
        == "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
    )
    assert args.raw_dataset_folder == "data\\raw"
    assert args.processed_dataset_folder == "data\\processed"
    assert args.logfile == "logs\\data_log.txt"
    assert args.loglevel == "debug"


def test_fetch_housing_data():
    housing_path = os.path.join("data\\raw", "housing")
    ingest_data.fetch_housing_data(
        "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz",
        housing_path,
    )
    tgz_path = os.path.join(housing_path, "housing.tgz")
    assert os.path.exists(tgz_path)
    csv_path = os.path.join(housing_path, "housing.csv")
    assert os.path.exists(csv_path)


def test_load_housing_data():
    housing_path = os.path.join("data\\raw", "housing")
    df = ingest_data.load_housing_data(housing_path)
    assert not df.empty
