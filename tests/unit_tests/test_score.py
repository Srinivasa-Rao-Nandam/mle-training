from src.regression import score


def test_parse_args():
    args = score.parse_arguments()
    assert args.test_data == "data\\processed\\housing\\test_data.csv"
    assert args.test_labels == "data\\processed\\housing\\test_label.csv"
    assert args.pickle_model == "artifacts\\best_model.pickle"
    assert args.logfile == "logs\\score_log.txt"
    assert args.loglevel == "debug"
