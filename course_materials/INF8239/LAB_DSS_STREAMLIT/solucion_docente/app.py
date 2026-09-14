from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dss import run_pipeline

st.set_page_config(
    page_title="INF-8239 · DSS territorial",
    page_icon="📊",
    layout="wide",
)

st.title("DSS para priorización territorial")
st.caption("Proyecto didáctico basado en RD DSS Hotspots · Edwin Ramón José Nolasco")

with st.sidebar:
    st.header("Configuración")
    uploaded = st.file_uploader("Dataset opcional", type=["csv"])
    top_k = st.slider("Cantidad de provincias", 3, 15, 10)
    execute = st.button("Ejecutar análisis", type="primary", use_container_width=True)


@st.cache_data(show_spinner=False)
def execute_pipeline(file_bytes: bytes | None) -> dict:
    return run_pipeline(file_bytes)


if not execute:
    st.info("Configure el análisis y pulse «Ejecutar análisis».")
    st.stop()

try:
    result = execute_pipeline(uploaded.getvalue() if uploaded else None)
except Exception as exc:
    st.error("No fue posible completar el análisis.")
    st.exception(exc)
    st.stop()

ranking = result["ranking"].head(top_k).copy()
metrics = result["metrics"].copy()

st.success(
    f"Análisis completado. Evaluación temporal: {result['test_year']}; "
    f"período priorizado: {result['latest_year']}."
)

m1, m2, m3 = st.columns(3)
m1.metric("Territorios analizados", len(result["ranking"]))
m2.metric("Alta prioridad", int((result["ranking"]["categoria"] == "Alta prioridad").sum()))
m3.metric("Mayor score", f"{result['ranking']['score_riesgo'].max():.2f}")

st.subheader("Comparación de modelos")
st.dataframe(metrics.style.format({"mae": "{:.2f}", "rmse": "{:.2f}", "r2": "{:.3f}"}))

st.subheader(f"Ranking Top-{top_k}")
st.dataframe(
    ranking,
    use_container_width=True,
    hide_index=True,
    column_config={
        "score_riesgo": st.column_config.ProgressColumn(
            "Score", min_value=0.0, max_value=1.0, format="%.2f"
        ),
        "prediccion": st.column_config.NumberColumn("Predicción", format="%.1f"),
    },
)

figure = px.bar(
    ranking.sort_values("score_riesgo"),
    x="score_riesgo",
    y="provincia",
    color="categoria",
    orientation="h",
    title="Prioridad relativa por territorio",
)
st.plotly_chart(figure, use_container_width=True)

province = st.selectbox("Consultar trazabilidad", ranking["provincia"])
detail = ranking[ranking["provincia"] == province].iloc[0]
st.markdown(f"**Categoría:** {detail['categoria']}")
st.markdown(f"**Regla aplicada:** {detail['regla']}")
st.markdown(f"**Recomendación:** {detail['recomendacion']}")
st.warning(
    "El sistema apoya la priorización; no sustituye la revisión de una persona responsable."
)

st.download_button(
    "Descargar ranking CSV",
    ranking.to_csv(index=False).encode("utf-8-sig"),
    "ranking_dss.csv",
    "text/csv",
)
