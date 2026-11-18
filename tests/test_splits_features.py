import numpy as np
import pandas as pd

from config import HOUSING_FEATURES, IRIS_FEATURES
from experiment import make_splits, prepare_seed
from features import add_features, feature_columns


def test_splits_disjoint():
    s = make_splits(506, 42)
    sets = {k: set(v.tolist()) for k, v in s.items()}
    assert len(sets["fit"] & sets["cal"]) == 0
    assert len(sets["fit"] & sets["val"]) == 0
    assert len(sets["fit"] & sets["test"]) == 0
    assert len(sets["cal"] & sets["test"]) == 0
    assert sum(len(v) for v in sets.values()) == 506


def test_prepare_seed_no_leakage_and_labels():
    p = prepare_seed("housing", 42)
    assert p["X"]["fit"].shape[0] == len(p["splits"]["fit"])
    assert p["X"]["test"].shape[0] == len(p["splits"]["test"])
    assert int(p["y_test"].sum()) == p["n_test_anom"]
    assert set(np.unique(p["y_test"])).issubset({0, 1})
    assert p["X"]["fit"].shape[1] == len(p["selected"])


def test_features_no_invalid_values():
    housing = pd.DataFrame(np.random.default_rng(0).normal(size=(20, len(HOUSING_FEATURES))),
                           columns=HOUSING_FEATURES)
    housing["RAD"] = np.arange(20) % 24
    housing["CHAS"] = np.arange(20) % 2
    out = add_features(housing, "housing")
    assert set(feature_columns("housing", "engineered")).issubset(out.columns)
    assert np.isfinite(out[feature_columns("housing", "engineered")].to_numpy()).all()

    iris = pd.DataFrame(np.abs(np.random.default_rng(1).normal(size=(20, len(IRIS_FEATURES)))) + 0.1,
                        columns=IRIS_FEATURES)
    out = add_features(iris, "iris")
    assert set(feature_columns("iris", "engineered")).issubset(out.columns)
    assert np.isfinite(out[feature_columns("iris", "engineered")].to_numpy()).all()
