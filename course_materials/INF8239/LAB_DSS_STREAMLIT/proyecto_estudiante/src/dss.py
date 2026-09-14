"""Núcleo del laboratorio. Complete cada TODO siguiendo la guía MoodleSafe."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

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
    raise NotImplementedError("Paso 05: implementar carga y normalización")


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Paso 05: implementar validación defensiva")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Paso 06: implementar variables temporales")


def temporal_split(trainable: pd.DataFrame):
    raise NotImplementedError("Paso 07: implementar separación temporal")


def train_and_compare(feature_df: pd.DataFrame):
    raise NotImplementedError("Pasos 08-09: baseline, candidato y métricas")


def classify_priority(score: float, delta_abs: float) -> Decision:
    raise NotImplementedError("Paso 10: implementar reglas DSS")


def build_ranking(model, feature_df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Paso 11: construir score y ranking")


def run_pipeline(source: str | Path | bytes | None = None) -> dict:
    raise NotImplementedError("Paso 12: integrar el pipeline")
