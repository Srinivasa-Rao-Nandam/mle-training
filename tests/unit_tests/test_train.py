from src.regression import train


def test_parse_args():
    args = train.parse_arguments()
    assert args.train_data == "data\\processed\\housing\\train_data.csv"
    assert args.train_labels == "data\\processed\\housing\\train_label.csv"
    assert args.pickle_output_folder == "artifacts"
    assert args.logfile == "logs\\train_log.txt"
    assert args.loglevel == "debug"
