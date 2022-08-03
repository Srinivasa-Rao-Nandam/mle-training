import argparse

import mlflow
import mlflow.sklearn

from ingest_data import transform_data
from score import score
from train import (
    MODEL_TYPE_DECISION_TREE,
    MODEL_TYPE_GRID_SEARCH_RANDOM_FOREST,
    MODEL_TYPE_LINEAR,
    MODEL_TYPE_RANDOM_SEARCH_RANDOM_FOREST,
    train,
)


def start_mlflow(
    exp_name,
    remote_server_uri,
    download_url,
    raw_dataset_folder,
    processed_dataset_folder,
    pickle_output_folder,
    model,
):
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="PARENT_RUN") as parent_run:
        mlflow.log_param("parent", "yes")

        with mlflow.start_run(run_name="CHILD_DATA_RUN", nested=True) as child_data_run:
            mlflow.log_param("child_data_run", "yes")
            train_data, train_labels, test_data, test_labels = transform_data(
                download_url, raw_dataset_folder, processed_dataset_folder
            )

        with mlflow.start_run(
            run_name="CHILD_TRAIN_RUN", nested=True
        ) as child_train_run:
            mlflow.log_param("child_train_run", "yes")
            model_path, _, _ = train(
                train_data, train_labels, pickle_output_folder, model
            )

        with mlflow.start_run(
            run_name="CHILD_SCORE_RUN", nested=True
        ) as child_score_run:
            mlflow.log_param("child_score_run", "yes")
            score(test_data, test_labels, model_path)

    print("parent run_id: {}".format(parent_run.info.run_id))
    print("child run_id : {}".format(child_data_run.info.run_id))
    print("child run_id : {}".format(child_score_run.info.run_id))
    print("child run_id : {}".format(child_train_run.info.run_id))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--exp_name",
        type=str,
        help="experiment name",
        default="housing_regression",
    )
    argparser.add_argument(
        "--remote_server_uri",
        type=str,
        help="uri of server",
        default="http://127.0.0.1:5000",
    )
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
        "--pickle_output_folder",
        type=str,
        help="Output folder for pickle",
        default="artifacts",
    )
    argparser.add_argument(
        "--model",
        type=str,
        help="type of model",
        default=MODEL_TYPE_GRID_SEARCH_RANDOM_FOREST,
        choices=[
            MODEL_TYPE_LINEAR,
            MODEL_TYPE_DECISION_TREE,
            MODEL_TYPE_RANDOM_SEARCH_RANDOM_FOREST,
            MODEL_TYPE_GRID_SEARCH_RANDOM_FOREST,
        ],
    )
    args = argparser.parse_args()
    start_mlflow(
        args.exp_name,
        args.remote_server_uri,
        args.download_url,
        args.raw_dataset_folder,
        args.processed_dataset_folder,
        args.pickle_output_folder,
        args.model,
    )
