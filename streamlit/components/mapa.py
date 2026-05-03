import streamlit as st
import pandas as pd
import requests
import plotly.express as px

@st.cache_data
def cargar_geojson_estados():
    url = "https://raw.githubusercontent.com/PhantomInsights/mexican-geojson/main/states.geojson"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data
def cargar_geojson_municipios():
    url = "https://raw.githubusercontent.com/PhantomInsights/mexican-geojson/main/municipalities.geojson"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def _centro_estado(df):
    if "LAT" in df.columns and "LON" in df.columns:
        return {"lat": df["LAT"].mean(), "lon": df["LON"].mean()}
    return {"lat": 23.6345, "lon": -102.5528}

def mapa_desaparecidos(data_raw: pd.DataFrame):

    # ── Limpia espacios en nombres de columnas ────────────────────────────────
    data_raw.columns = data_raw.columns.str.strip()

    # ── Muestra columnas disponibles (debug, borra después) ───────────────────
    st.write("Columnas disponibles:", data_raw.columns.tolist())
    st.write("Valores únicos de ESTATUS_VICTIMA:", 
             data_raw["ESTATUS_VICTIMA"].unique().tolist())

    # ── Filtra solo personas desaparecidas ────────────────────────────────────
    # Ajusta los valores según lo que veas en el debug de arriba
    ESTATUS_DESAPARECIDO = ["DESAPARECIDO", "DESAPARECIDA"]   # ← ajusta si difiere

    df_desap = data_raw[
        data_raw["ESTATUS_VICTIMA"].str.upper().str.strip().isin(ESTATUS_DESAPARECIDO)
    ].copy()

    st.write(f"Total registros desaparecidos: {len(df_desap):,}")

    # ── Filtros en sidebar ────────────────────────────────────────────────────
    estados_ordenados = sorted(df_desap["ENTIDAD"].dropna().unique().tolist())
    estado_sel = st.sidebar.selectbox(
        "1. Selecciona un Estado:",
        ["Todos"] + estados_ordenados,
        key="filtro_estado"
    )

    if estado_sel == "Todos":
        df_filtrado = df_desap.copy()
    else:
        df_filtrado = df_desap[df_desap["ENTIDAD"] == estado_sel].copy()

    municipios_disponibles = sorted(df_filtrado["MUNICIPIO"].dropna().unique().tolist())
    municipio_sel = st.sidebar.selectbox(
        "2. Selecciona un Municipio:",
        ["Todos"] + municipios_disponibles,
        key="filtro_municipio"
    )

    if municipio_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado["MUNICIPIO"] == municipio_sel].copy()

    # ── Contar desaparecidos por zona geográfica ──────────────────────────────
    if estado_sel == "Todos":
        # Nivel PAÍS → agrupa por estado
        # Ajusta "CVE_ENT" al nombre exacto de tu columna de clave de estado
        df_mapa = (
            df_filtrado
            .groupby(["ENTIDAD", "CVE_ENT"], as_index=False)
            .size()
            .rename(columns={"size": "TOTAL_DESAPARECIDOS"})
        )
        col_location = "CVE_ENT"
        feature_key  = "properties.CVEGEO"
        zoom_inicial = 4.5
        centro       = {"lat": 23.6345, "lon": -102.5528}
        geojson_data = cargar_geojson_estados()
        hover_name   = "ENTIDAD"

    else:
        # Nivel ESTADO → agrupa por municipio
        # Ajusta "CVE_MUN" al nombre exacto de tu columna de clave de municipio
        df_mapa = (
            df_filtrado
            .groupby(["MUNICIPIO", "CVE_MUN"], as_index=False)
            .size()
            .rename(columns={"size": "TOTAL_DESAPARECIDOS"})
        )
        col_location = "CVE_MUN"
        feature_key  = "properties.CVEGEO"
        zoom_inicial = 7
        centro       = _centro_estado(df_filtrado)
        geojson_data = cargar_geojson_municipios()
        hover_name   = "MUNICIPIO"

    # ── Métricas ──────────────────────────────────────────────────────────────
    st.write("### 🗺️ Mapa de Personas Desaparecidas")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total desaparecidos",  f"{df_mapa['TOTAL_DESAPARECIDOS'].sum():,}")
    c2.metric("Zonas mostradas",       len(df_mapa))
    c3.metric("Máximo por zona",       f"{df_mapa['TOTAL_DESAPARECIDOS'].max():,}")

    # ── Mapa ──────────────────────────────────────────────────────────────────
    try:
        fig = px.choropleth_mapbox(
            df_mapa,
            geojson=geojson_data,
            locations=col_location,
            featureidkey=feature_key,
            color="TOTAL_DESAPARECIDOS",
            color_continuous_scale=["#FFF5F0","#FEB69B","#FC6D43","#DE2D26","#67000D"],
            range_color=(0, df_mapa["TOTAL_DESAPARECIDOS"].max()),
            mapbox_style="carto-positron",
            zoom=zoom_inicial,
            center=centro,
            opacity=0.75,
            hover_name=hover_name,
            hover_data={col_location: False, "TOTAL_DESAPARECIDOS": True},
            labels={"TOTAL_DESAPARECIDOS": "Desaparecidos"},
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error al renderizar el mapa: {e}")
        st.dataframe(df_mapa)