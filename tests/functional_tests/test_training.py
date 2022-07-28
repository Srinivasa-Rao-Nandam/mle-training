from src.regression import train


def test_train():
    model = train.train()
    assert model is not None


if __name__ == "__main__":
    test_train()
