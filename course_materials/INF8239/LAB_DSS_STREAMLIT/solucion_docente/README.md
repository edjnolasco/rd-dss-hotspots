# INF-8239 · Laboratorio DSS con Streamlit

Solución docente del laboratorio **DSS para priorización territorial basada en riesgo**.

Autor: **Edwin Ramón José Nolasco**

## Preparación

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

La aplicación descarga por defecto el dataset del repositorio `rd-dss-hotspots`.
También admite un CSV con las columnas `provincia`, `year` y `fallecidos`.

## Flujo

Datos → características → evaluación temporal → modelo → score → reglas → ranking → decisión.

## Uso responsable

La salida prioriza casos para revisión humana. No constituye por sí sola una orden de intervención ni sustituye el juicio del responsable institucional.
