# app.py
import sys
import os


# 1. Obtener la ruta absoluta del directorio actual (la carpeta 'streamlit')
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Obtener la ruta del directorio padre (la raíz del proyecto)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# 3. Agregar el directorio padre al "Path" temporal de Python
sys.path.append(parent_dir)
from components.mapa import mapa_desaparecidos
import streamlit as st
import os
import plotly.graph_objects as go
from config.settings import settings
from pages.forecasting import panel_metricas, matriz_confusion_sin_confidencialidad
from src.models.forecasting import main_forecasting
from components.matriz_confunsion import matriz_confusion_con_etiquetas
import plotly.express as px
import pandas as pd
import numpy as np


# ─── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title=settings.APP_NAME,
    page_icon=settings.APP_ICON,
    layout=settings.LAYOUT,           # "centered" | "wide"
    initial_sidebar_state="expanded", # "auto" | "expanded" | "collapsed"
    menu_items={
        "Get Help": "https://docs.miapp.com",
        "Report a bug": "mailto:soporte@miapp.com",
        "About": f"## {settings.APP_NAME}\nVersión {settings.VERSION}",
    },
)

# ─── CSS global (opcional) ────────────────────────────────────────────────────
def load_css():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_css = os.path.join(directorio_actual, "assets", "styles.css")
    
    # Agregamos la codificación utf-8 aquí 👇
    with open(ruta_css, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ─── Página principal ─────────────────────────────────────────────────────────
def main():
    st.title("Panel de Control de Análisis de Datos de Personas Desaparecidas")
    st.subheader("Database: Secretariado.csx")
    y_test, y_predictions, model, X_test, data_raw = main_forecasting()
    panel_metricas(y_test, y_predictions)
    col1, col2 = st.columns(2)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    with col1:
        conteos = data_raw["ESTATUS_VICTIMA"].str.upper().str.strip().value_counts()

        fig = go.Figure(data=go.Pie(
            labels=conteos.index.tolist(),
            values=conteos.values.tolist(),
            hole=0.4,
            marker=dict(colors=["#d62728", "#1f77b4", "#ff7f0e"]),
        ))
        fig.update_layout(title="Distribución de Estatus de Víctimas")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        matriz_confusion_con_etiquetas(y_test, y_predictions)

main()