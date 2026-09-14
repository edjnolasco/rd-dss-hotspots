import pandas as pd
import pytest

from src.dss import (
    build_ranking,
    classify_priority,
    create_features,
    normalize_data,
    run_pipeline,
    temporal_split,
    train_and_compare,
)


@pytest.fixture
def sample_data():
    rows = []
    values = {
        "Azua": [10, 12, 15, 18, 20],
        "Duarte": [30, 29, 35, 40, 44],
        "Samaná": [8, 7, 9, 8, 11],
    }
    for province, deaths in values.items():
        for year, value in zip(range(2020, 2025), deaths):
            rows.append({"provincia": province, "year": year, "fallecidos": value})
    return pd.DataFrame(rows)


def test_normalize_rejects_missing_columns():
    with pytest.raises(ValueError, match="Faltan columnas"):
        normalize_data(pd.DataFrame({"provincia": ["Azua"]}))


def test_create_features_builds_lags_and_target(sample_data):
    result = create_features(sample_data)
    assert {"fallecidos_prev_1", "rolling_mean_3", "target_next"} <= set(result)
    assert result["target_next"].notna().sum() == 12


def test_temporal_split_uses_latest_trainable_year(sample_data):
    features = create_features(sample_data).dropna(subset=["target_next"])
    train, test, year = temporal_split(features)
    assert train["year"].max() < year
    assert test["year"].nunique() == 1


def test_rule_engine_returns_traceable_decision():
    decision = classify_priority(0.90, 5)
    assert decision.categoria == "Alta prioridad"
    assert decision.regla.startswith("R1")
    assert decision.recomendacion


def test_ranking_is_sorted(sample_data):
    features = create_features(sample_data)
    model, _, _ = train_and_compare(features)
    ranking = build_ranking(model, features)
    assert ranking["score_riesgo"].is_monotonic_decreasing
    assert ranking["regla"].notna().all()


def test_pipeline_returns_required_outputs(sample_data, tmp_path):
    path = tmp_path / "sample.csv"
    sample_data.to_csv(path, index=False)
    result = run_pipeline(path)
    assert {"metrics", "ranking", "test_year", "latest_year"} <= set(result)
    assert len(result["ranking"]) == 3
