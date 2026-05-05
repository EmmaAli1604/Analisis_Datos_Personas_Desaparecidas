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
    st.markdown(
        "Ya no se pudo hacer el dashboard, ya fue todo, ya que termine el semestre. ")

main()