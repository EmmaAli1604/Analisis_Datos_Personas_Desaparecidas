import streamlit as st
from config.lecture import load_lecture_data_raw, load_lecture_data_inputed, load_lecture_data_processed
from components.layout import page_layout

def render_dataframe(df, label: str):
    """Muestra métricas + dataframe de forma consistente."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas",     df.shape[0])
    col2.metric("Columnas",  df.shape[1])
    col3.metric("Nulos",     df.isnull().sum().sum())

    st.dataframe(df, use_container_width=True, height=450)

    st.download_button(
        label=f"⬇️ Descargar {label} (.csv)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{label.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

def main():
    page_layout("🗄️ Base de Datos")
    st.markdown("Selecciona qué versión de los datos quieres explorar.")

    # ─── Menú de selección ────────────────────────────────────────────────────
    OPCIONES = {
        "🟠 Raw — Datos originales":       "raw",
        "🟢 Imputados — Nulos tratados":   "imputed",
        "🟡 Procesados — Listos para uso": "processed",
    }

    seleccion = st.radio(
        label="Versión de los datos",
        options=list(OPCIONES.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ─── Carga lazy según selección ───────────────────────────────────────────
    vista = OPCIONES[seleccion]

    with st.spinner("Cargando datos..."):
        if vista == "raw":
            df    = load_lecture_data_raw()
            label = "Raw"
            st.info(
                "**Datos originales** sin ningún tipo de transformación. "
                "Pueden contener nulos, duplicados o valores atípicos.",
                icon="🟠",
            )
        elif vista == "processed":
            df    = load_lecture_data_processed()
            label = "Procesados"
            st.info(
                "**Datos procesados**: limpios, normalizados y donde ha datos nulos pero se han dejado como tal para análisis posteriores.",
                icon="🟡",
            )
        else:  # imputed
            df    = load_lecture_data_inputed()
            label = "Imputados"
            st.info(
                "**Datos imputados**: los valores nulos han sido reemplazados "
                "mediante técnicas de imputación estadística.",
                icon="🟢",
            )

    render_dataframe(df, label)

main()