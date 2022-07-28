from ast import Import


def test_install():
    try:
        import pandas
        import sklearn
    except ImportError:
        assert False
    assert True
