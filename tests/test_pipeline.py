from pathlib import Path
import sys

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import FEATURES, TARGET, build_pipeline, load_dataset


def test_pipeline_trains_predicts_and_serializes(tmp_path: Path):
    data = load_dataset()
    labeled = data.loc[data[TARGET].notna()]
    X, y = labeled[FEATURES], labeled[TARGET].astype(int)
    pipeline = build_pipeline().fit(X.iloc[:1200], y.iloc[:1200])
    sample = X.iloc[1200:1210].copy()
    predictions = pipeline.predict(sample)
    assert len(predictions) == 10
    assert set(predictions).issubset({0, 1})

    artifact = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, artifact)
    restored = joblib.load(artifact)
    assert list(restored.predict(sample)) == list(predictions)


def test_pipeline_handles_missing_values_and_unseen_categories():
    data = load_dataset()
    labeled = data.loc[data[TARGET].notna()]
    X, y = labeled[FEATURES], labeled[TARGET].astype(int)
    pipeline = build_pipeline().fit(X.iloc[:1200], y.iloc[:1200])
    sample = X.iloc[[1200]].copy()
    sample.loc[sample.index[0], "Region"] = "Unseen region"
    sample.loc[sample.index[0], "Rainfall_mm"] = float("nan")
    prediction = pipeline.predict(sample)
    assert len(prediction) == 1
    assert prediction[0] in {0, 1}
