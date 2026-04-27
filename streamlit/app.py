# app.py
import streamlit as st
import os
from config.settings import settings

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

# ─── Estado global inicial ────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "usuario": None,
        "autenticado": False,
        "tema": "light",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.image("assets/images/logo.png", use_container_width=True)
        st.markdown("---")

        st.markdown("### Navegación")
        st.page_link("app.py",                    label="🏠 Inicio")
        st.page_link("pages/01_dashboard.py",     label="📊 Dashboard")
        st.page_link("pages/02_analisis.py",      label="🔍 Análisis")
        st.page_link("pages/03_configuracion.py", label="⚙️ Configuración")

        st.markdown("---")
        if st.session_state.autenticado:
            st.caption(f"👤 {st.session_state.usuario}")
            if st.button("Cerrar sesión", use_container_width=True):
                st.session_state.autenticado = False
                st.session_state.usuario = None
                st.rerun()

render_sidebar()

# ─── Página principal ─────────────────────────────────────────────────────────
def main():
    st.title("🏠 Bienvenido")
    st.markdown("Selecciona una sección en el menú lateral para comenzar.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 **Dashboard**\nVisualiza métricas clave.")
    with col2:
        st.info("🔍 **Análisis**\nExplora tus datos.")
    with col3:
        st.info("⚙️ **Configuración**\nAjusta parámetros.")

main()