from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_URL = (
    "https://raw.githubusercontent.com/edjnolasco/rd-dss-hotspots/"
    "main/data/fallecimientos_provincias.csv"
)
FEATURES = [
    "fallecidos",
    "fallecidos_prev_1",
    "fallecidos_prev_2",
    "delta_abs",
    "rolling_mean_3",
]


@dataclass(frozen=True)
class Decision:
    categoria: str
    regla: str
    recomendacion: str


def load_data(source: str | Path | bytes | None = None) -> pd.DataFrame:
    """Carga el CSV oficial del laboratorio, una ruta local o bytes subidos."""
    if source is None:
        df = pd.read_csv(DATA_URL)
    elif isinstance(source, bytes):
        df = pd.read_csv(BytesIO(source))
    else:
        df = pd.read_csv(source)
    return normalize_data(df)


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {"provincia", "year", "fallecidos"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    out = df[list(required)].copy()
    out["provincia"] = (
        out["provincia"].astype(str).str.strip().str.lower().str.title()
    )
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["fallecidos"] = pd.to_numeric(out["fallecidos"], errors="coerce")
    out = out.dropna(subset=["provincia", "year", "fallecidos"])
    out["year"] = out["year"].astype(int)
    out["fallecidos"] = out["fallecidos"].astype(float)
    out = out.drop_duplicates(["provincia", "year"], keep="last")
    out = out.sort_values(["provincia", "year"]).reset_index(drop=True)

    if out.empty:
        raise ValueError("El dataset no contiene observaciones válidas.")
    if out["year"].nunique() < 3:
        raise ValueError("Se requieren al menos tres períodos.")
    if (out["fallecidos"] < 0).any():
        raise ValueError("La variable fallecidos no admite valores negativos.")
    return out


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_data(df)
    group = out.groupby("provincia", group_keys=False)["fallecidos"]
    out["fallecidos_prev_1"] = group.shift(1)
    out["fallecidos_prev_2"] = group.shift(2)
    out["delta_abs"] = out["fallecidos"] - out["fallecidos_prev_1"]
    out["rolling_mean_3"] = (
        group.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    out["target_next"] = group.shift(-1)
    return out


def temporal_split(
    trainable: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    years = sorted(trainable["year"].unique())
    if len(years) < 2:
        raise ValueError("No hay períodos suficientes para una evaluación temporal.")
    test_year = int(years[-1])
    train = trainable[trainable["year"] < test_year].copy()
    test = trainable[trainable["year"] == test_year].copy()
    if train.empty or test.empty:
        raise ValueError("La separación temporal produjo un conjunto vacío.")
    return train, test, test_year


def _metrics(model, test: pd.DataFrame) -> dict[str, float]:
    prediction = model.predict(test[FEATURES])
    return {
        "mae": float(mean_absolute_error(test["target_next"], prediction)),
        "rmse": float(
            np.sqrt(mean_squared_error(test["target_next"], prediction))
        ),
        "r2": float(r2_score(test["target_next"], prediction)),
    }


def train_and_compare(
    feature_df: pd.DataFrame,
) -> tuple[RandomForestRegressor, pd.DataFrame, int]:
    trainable = feature_df.dropna(subset=["target_next"]).copy()
    trainable[FEATURES] = trainable[FEATURES].fillna(0.0)
    train, test, test_year = temporal_split(trainable)

    baseline = DummyRegressor(strategy="mean")
    candidate = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    baseline.fit(train[FEATURES], train["target_next"])
    candidate.fit(train[FEATURES], train["target_next"])

    rows = [
        {"modelo": "Baseline promedio", **_metrics(baseline, test)},
        {"modelo": "Random Forest", **_metrics(candidate, test)},
    ]
    metrics = pd.DataFrame(rows)

    final_model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(trainable[FEATURES], trainable["target_next"])
    return final_model, metrics, test_year


def classify_priority(score: float, delta_abs: float) -> Decision:
    if score >= 0.80:
        return Decision(
            "Alta prioridad",
            "R1: score >= 0.80",
            "Priorizar revisión e intervención; validar recursos disponibles.",
        )
    if score >= 0.60 and delta_abs > 0:
        return Decision(
            "Vigilancia preventiva",
            "R2: score >= 0.60 y tendencia creciente",
            "Monitorear la evolución y preparar medidas preventivas.",
        )
    if score >= 0.40:
        return Decision(
            "Atención moderada",
            "R3: 0.40 <= score < 0.80",
            "Mantener seguimiento periódico y revisar nuevos datos.",
        )
    return Decision(
        "Seguimiento rutinario",
        "R4: score < 0.40",
        "Conservar observación y reevaluar en el próximo ciclo.",
    )


def build_ranking(model, feature_df: pd.DataFrame) -> pd.DataFrame:
    latest_year = int(feature_df["year"].max())
    latest = feature_df[feature_df["year"] == latest_year].copy()
    latest[FEATURES] = latest[FEATURES].fillna(0.0)
    latest["prediccion"] = np.maximum(model.predict(latest[FEATURES]), 0.0)

    low, high = latest["prediccion"].min(), latest["prediccion"].max()
    latest["score_riesgo"] = (
        0.5 if high == low else (latest["prediccion"] - low) / (high - low)
    )

    decisions = [
        classify_priority(float(score), float(delta or 0.0))
        for score, delta in zip(
            latest["score_riesgo"], latest["delta_abs"].fillna(0.0)
        )
    ]
    latest["categoria"] = [item.categoria for item in decisions]
    latest["regla"] = [item.regla for item in decisions]
    latest["recomendacion"] = [item.recomendacion for item in decisions]
    return latest[
        [
            "provincia",
            "year",
            "fallecidos",
            "prediccion",
            "score_riesgo",
            "categoria",
            "regla",
            "recomendacion",
        ]
    ].sort_values("score_riesgo", ascending=False).reset_index(drop=True)


def run_pipeline(source: str | Path | bytes | None = None) -> dict:
    data = load_data(source)
    features = create_features(data)
    model, metrics, test_year = train_and_compare(features)
    ranking = build_ranking(model, features)
    return {
        "data": data,
        "features": features,
        "model": model,
        "metrics": metrics,
        "ranking": ranking,
        "test_year": test_year,
        "latest_year": int(features["year"].max()),
    }
