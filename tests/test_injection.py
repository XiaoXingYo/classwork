import numpy as np
import pandas as pd

from inject import inject_anomalies, reference_stats
from config import HOUSING_FEATURES


def test_injection_manifest_and_labels():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(100, len(HOUSING_FEATURES))), columns=HOUSING_FEATURES)
    stats = reference_stats(df, HOUSING_FEATURES)
    res = inject_anomalies(df, 8, 42, HOUSING_FEATURES, stats)
    assert res.labels.sum() == 8
    assert len(res.manifest) == 8
    assert set(res.manifest["injection_type"]).issubset({"point", "multi", "extreme"})
    changed = 0
    for _, row in res.manifest.iterrows():
        pos = int(row["row_position"])
        changed += int(not np.allclose(res.frame.iloc[pos].to_numpy(), df.iloc[pos].to_numpy()))
    assert changed == 8


def test_injection_reproducible():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(80, len(HOUSING_FEATURES))), columns=HOUSING_FEATURES)
    stats = reference_stats(df, HOUSING_FEATURES)
    a = inject_anomalies(df, 6, 7, HOUSING_FEATURES, stats)
    b = inject_anomalies(df, 6, 7, HOUSING_FEATURES, stats)
    assert a.manifest.equals(b.manifest)
    assert np.allclose(a.frame.to_numpy(), b.frame.to_numpy())
