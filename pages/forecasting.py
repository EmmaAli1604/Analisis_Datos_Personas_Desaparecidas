import streamlit as st
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from src.models.forecasting import main_forecasting
from src.models.forecasting import auditoria_columnas
from src.models.normalization import normaliza_data
from components.layout import page_layout
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score,
)
from prophet import Prophet
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from src.models.normalization import normaliza_data
import plotly.express as px
from scipy.stats import linregress


def visualizar_importancia_plotly(model, X_train):
    importancias = model.feature_importances_
    df_imp = pd.DataFrame({
        'Variable': X_train.columns,
        'Importancia': importancias
    }).sort_values(by='Importancia', ascending=True)

    fig = px.bar(df_imp, x='Importancia', y='Variable', orientation='h',
                title='Peso de las Variables en la Predicción en ROC')
    
    # En Streamlit se usa esta función específica
    st.plotly_chart(fig, use_container_width=True)

def plot_roc_curve(model, X_test, y_test):
    # 1. Obtener las clases del modelo y calcular probabilidades
    classes = model.classes_ 
    y_prob = model.predict_proba(X_test) 
    
    # 2. Binarizar y_test correctamente
    # Aquí es donde fallaba porque y_test recibía accidentalmente y_prob
    y_bin = label_binarize(y_test, classes=classes)

    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e"]

    for i, clase in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{clase} (AUC = {roc_auc:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatorio", line=dict(color="gray", dash="dash")))

    fig.update_layout(title="Curva ROC Corregida", template="plotly_white")
    return fig

def plot_serie_tiempo(X_test, y_test, y_pred, data_original):
    clases  = ["DESAPARECIDA", "NO LOCALIZADA"]
    colores = {"DESAPARECIDA": "#1f77b4", "NO LOCALIZADA": "#ff7f0e", "CONFIDENCIAL": "#d62728"}

    # ── Datos históricos completos ─────────────────────────────────────────
    df_hist = data_original[["FECHA_DESAPARICION", "ESTATUS_VICTIMA"]].copy()
    df_hist["FECHA_DESAPARICION"] = pd.to_datetime(df_hist["FECHA_DESAPARICION"], errors="coerce")
    df_hist["ANIO"] = df_hist["FECHA_DESAPARICION"].dt.year
    hist_grouped = (
        df_hist.dropna(subset=["ANIO"])
        .groupby(["ANIO", "ESTATUS_VICTIMA"])
        .size()
        .reset_index(name="CONTEO")
    )

    # ── Separar CONFIDENCIAL en: con fecha válida vs sin fecha ────────────
    df_conf = df_hist[df_hist["ESTATUS_VICTIMA"] == "CONFIDENCIAL"].copy()

    ANIO_SOSPECHOSO = df_conf["ANIO"].value_counts().idxmax()  # el año pico = fecha falsa
    df_conf_valida   = df_conf[df_conf["ANIO"] != ANIO_SOSPECHOSO]
    n_conf_sin_fecha = (df_conf["ANIO"] == ANIO_SOSPECHOSO).sum()

    # Años de referencia para distribuir la incertidumbre
    anios_referencia = sorted(df_hist["ANIO"].dropna().unique())
    anios_referencia = [a for a in anios_referencia if 1990 <= a <= 2026]

    # Distribuir n_conf_sin_fecha proporcionalmente a tendencia histórica total
    total_por_anio = (
        hist_grouped.groupby("ANIO")["CONTEO"].sum()
        .reindex(anios_referencia, fill_value=0)
    )
    pesos = total_por_anio / total_por_anio.sum()
    distribucion_conf = (pesos * n_conf_sin_fecha).round().astype(int)

    # CONFIDENCIAL con fecha válida agrupada por año
    conf_valida_grouped = (
        df_conf_valida.groupby("ANIO").size()
        .reindex(anios_referencia, fill_value=0)
    )

    # Centro de la banda = casos válidos + distribución estimada
    conf_centro = conf_valida_grouped + distribucion_conf
    # Banda de incertidumbre: ±30% de los casos distribuidos
    margen = (distribucion_conf * 0.30).round().astype(int)
    conf_upper = conf_centro + margen
    conf_lower = (conf_centro - margen).clip(lower=0)

    # ── Datos predichos ────────────────────────────────────────────────────
    col_anio = "FECHA_DESAPARICION_ANIO"
    if col_anio not in X_test.columns:
        st.warning(f"No se encontró '{col_anio}' en X_test")
        return go.Figure()

    factor_escala = 1 / 0.2
    df_pred = pd.DataFrame({
        "ANIO"        : X_test[col_anio].values,
        "ESTATUS_PRED": y_pred,
    })
    df_pred = df_pred[df_pred["ANIO"] > 1900]
    pred_grouped = (
        df_pred.groupby(["ANIO", "ESTATUS_PRED"])
        .size()
        .reset_index(name="CONTEO")
    )
    pred_grouped["CONTEO"] = (pred_grouped["CONTEO"] * factor_escala).round()

    # ── Figura ─────────────────────────────────────────────────────────────
    fig = go.Figure()

    # Banda de incertidumbre CONFIDENCIAL (primero para que quede atrás)
    anios_list = list(anios_referencia)
    fig.add_trace(go.Scatter(
        x=anios_list + anios_list[::-1],
        y=conf_upper.tolist() + conf_lower.tolist()[::-1],
        fill="toself",
        fillcolor="rgba(214, 39, 40, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="CONFIDENCIAL (incertidumbre estimada)",
    ))

    # Línea central CONFIDENCIAL estimada
    fig.add_trace(go.Scatter(
        x=anios_list,
        y=conf_centro.tolist(),
        mode="lines+markers",
        name="CONFIDENCIAL (estimado)",
        line=dict(color=colores["CONFIDENCIAL"], width=2, dash="dash"),
        marker=dict(size=4),
    ))

    # Línea real de CONFIDENCIAL con fecha válida
    df_conf_real = hist_grouped[hist_grouped["ESTATUS_VICTIMA"] == "CONFIDENCIAL"].sort_values("ANIO")
    df_conf_real = df_conf_real[df_conf_real["ANIO"] != ANIO_SOSPECHOSO]
    if not df_conf_real.empty:
        fig.add_trace(go.Scatter(
            x=df_conf_real["ANIO"], y=df_conf_real["CONTEO"],
            mode="lines+markers",
            name="CONFIDENCIAL (fecha real)",
            line=dict(color=colores["CONFIDENCIAL"], width=2, dash="solid"),
            marker=dict(size=5),
        ))

    # DESAPARECIDA y NO LOCALIZADA (real + predicho)
    for clase in clases:
        color = colores[clase]
        df_c = hist_grouped[hist_grouped["ESTATUS_VICTIMA"] == clase].sort_values("ANIO")
        if not df_c.empty:
            fig.add_trace(go.Scatter(
                x=df_c["ANIO"], y=df_c["CONTEO"],
                mode="lines+markers",
                name=f"{clase} (real)",
                line=dict(color=color, width=2, dash="solid"),
                marker=dict(size=5),
            ))

        df_p = pred_grouped[pred_grouped["ESTATUS_PRED"] == clase].sort_values("ANIO")
        if not df_p.empty:
            fig.add_trace(go.Scatter(
                x=df_p["ANIO"], y=df_p["CONTEO"],
                mode="lines+markers",
                name=f"{clase} (predicho)",
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=5, symbol="diamond"),
            ))

    anio_min = 1990
    anio_max = int(df_hist["ANIO"].dropna().max()) + 1

    fig.update_layout(
        title="Serie de Tiempo — Casos por Año y Estatus",
        xaxis_title="Año",
        yaxis_title="Número de Casos",
        legend_title="Estatus",
        hovermode="x unified",
        xaxis=dict(range=[anio_min, anio_max], tickangle=-45, dtick=1),
        height=500,
    )

    return fig

def plot_forecast_prophet(data_original, clases=["DESAPARECIDA", "NO LOCALIZADA"], anios_forecast=10):
    """
    Genera un forecast de 10 años con Prophet para cada clase,
    agrupando previamente a nivel mensual para mayor precisión.
    """
    df = data_original[["FECHA_DESAPARICION", "ESTATUS_VICTIMA"]].copy()
    df["FECHA_DESAPARICION"] = pd.to_datetime(df["FECHA_DESAPARICION"], errors="coerce")
    df = df.dropna(subset=["FECHA_DESAPARICION"])

    # Excluir fechas futuras o absurdas
    hoy = pd.Timestamp.today()
    df = df[(df["FECHA_DESAPARICION"] <= hoy) & (df["FECHA_DESAPARICION"].dt.year >= 1990)]

    colores = {
        "DESAPARECIDA"  : {"linea": "#1f77b4", "banda": "rgba(31,119,180,0.2)"},
        "NO LOCALIZADA" : {"linea": "#ff7f0e", "banda": "rgba(255,127,14,0.2)"},
    }

    # Como ahora la frecuencia será mensual, calculamos los meses a futuro
    meses_forecast = anios_forecast * 12
    fig = go.Figure()

    # Fecha de corte global para la línea vertical
    fecha_corte_global = df["FECHA_DESAPARICION"].max()

    for clase in clases:
        color = colores.get(clase, {"linea": "#6B35AC", "banda": "rgba(107,53,172,0.2)"})

        # ── 1. Preparar serie MENSUAL para Prophet ─────────────────────────
        df_clase = df[df["ESTATUS_VICTIMA"] == clase].copy()
        
        # Redondear fechas al inicio del mes
        df_clase["ds"] = df_clase["FECHA_DESAPARICION"].dt.to_period("M").dt.to_timestamp()
        
        df_clase_mensual = (
            df_clase.groupby("ds")
            .size()
            .reset_index(name="y")
        )

        if df_clase_mensual.empty:
            continue

        # Rellenar meses sin registros con 0
        rango_completo = pd.date_range(
            start=df_clase_mensual["ds"].min(),
            end=df_clase_mensual["ds"].max(),
            freq="MS" # Month Start
        )
        
        df_clase_mensual = (
            df_clase_mensual.set_index("ds")
            .reindex(rango_completo, fill_value=0)
            .reset_index()
            .rename(columns={"index": "ds"})
        )

        # ── 2. Entrenar Prophet ────────────────────────────────────────────
        modelo = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,   # Desactivado porque los datos son mensuales
            daily_seasonality=False,    # Desactivado por la misma razón
            changepoint_prior_scale=0.1, # Aumentado un poco (0.1) para adaptarse mejor a los cambios de tendencia recientes
            interval_width=0.95,         
        )
        modelo.fit(df_clase_mensual)

        # ── 3. Forecast ────────────────────────────────────────────────────
        futuro = modelo.make_future_dataframe(periods=meses_forecast, freq="MS")
        pronostico = modelo.predict(futuro)

        # Clip: no permitir valores negativos
        pronostico["yhat"]       = pronostico["yhat"].clip(lower=0)
        pronostico["yhat_lower"] = pronostico["yhat_lower"].clip(lower=0)
        pronostico["yhat_upper"] = pronostico["yhat_upper"].clip(lower=0)

        # Separar histórico y futuro
        fecha_corte = df_clase_mensual["ds"].max()
        forecast_futuro = pronostico[pronostico["ds"] > fecha_corte]
        
        # ── Banda de incertidumbre (futuro) ────────────────────────────────
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_futuro["ds"], forecast_futuro["ds"].iloc[::-1]]),
            y=pd.concat([forecast_futuro["yhat_upper"], forecast_futuro["yhat_lower"].iloc[::-1]]),
            fill="toself",
            fillcolor=color["banda"],
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ))

        # ── Línea histórica real ───────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=df_clase_mensual["ds"],
            y=df_clase_mensual["y"],
            mode="lines",
            name=f"{clase} (histórico)",
            line=dict(color=color["linea"], width=2),
        ))

        # ── Línea forecast (futuro) ────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=forecast_futuro["ds"],
            y=forecast_futuro["yhat"],
            mode="lines",
            name=f"{clase} (forecast 10 años)",
            line=dict(color=color["linea"], width=2, dash="dot"),
        ))

    # Línea vertical en fecha de corte global
    if not df.empty:
        fig.add_vline(
            x=fecha_corte_global.timestamp() * 1000,
            line_dash="dash",
            line_color="gray",
            annotation_text="Hoy",
            annotation_position="top right",
        )

    fig.update_layout(
        title="Predicción a 10 años — Casos por Estatus (Modelo Mensual)",
        xaxis_title="Fecha",
        yaxis_title="Casos por Mes",
        legend_title="Estatus",
        hovermode="x unified",
        height=520,
        xaxis=dict(tickangle=-45),
    )

    return fig

def plot_forecast_rf(model, data_original, anios_forecast=10, n_simulaciones=3):
    """
    Forecast RF por muestreo de perfiles históricos con proyección de tendencia de volumen.
    """
    colores = {
        "DESAPARECIDA"  : {"linea": "#1f77b4", "banda": "rgba(31,119,180,0.15)"},
        "NO LOCALIZADA" : {"linea": "#ff7f0e", "banda": "rgba(255,127,14,0.15)"},
    }
    
    features_modelo = model.feature_names_in_
    features_fecha  = [c for c in features_modelo if "FECHA_DESAPARICION" in c]
    features_perfil = [c for c in features_modelo if c not in features_fecha]

    # ── 1. Preparar histórico real ─────────────────────────────────────────
    df_hist = data_original[["FECHA_DESAPARICION", "ESTATUS_VICTIMA"]].copy()
    df_hist["FECHA_DESAPARICION"] = pd.to_datetime(df_hist["FECHA_DESAPARICION"], errors="coerce")
    df_hist = df_hist.dropna(subset=["FECHA_DESAPARICION"])
    df_hist = df_hist[df_hist["ESTATUS_VICTIMA"].isin(["DESAPARECIDA", "NO LOCALIZADA"])]
    df_hist = df_hist[df_hist["FECHA_DESAPARICION"].dt.year >= 1990]

    # Agrupar histórico por mes
    df_hist["MES"] = df_hist["FECHA_DESAPARICION"].dt.to_period("M").dt.to_timestamp()
    hist_mensual = df_hist.groupby(["MES", "ESTATUS_VICTIMA"]).size().reset_index(name="CONTEO")

    # ── 2. Calcular Tendencia de Volumen (Para no tener una línea plana) ──
    fecha_ancla = pd.Timestamp.today() - pd.DateOffset(years=3) # Usamos últimos 3 años para la tendencia
    df_tendencia = hist_mensual[hist_mensual["MES"] >= fecha_ancla]
    
    vol_total_mes = df_tendencia.groupby("MES")["CONTEO"].sum().reset_index()
    vol_total_mes["MES_NUM"] = np.arange(len(vol_total_mes))
    
    # Regresión lineal simple para saber cuántos casos se suman/restan por mes
    if len(vol_total_mes) > 2:
        slope, intercept, _, _, _ = linregress(vol_total_mes["MES_NUM"], vol_total_mes["CONTEO"])
    else:
        slope = 0 # Fallback si no hay datos
        
    volumen_base_actual = vol_total_mes["CONTEO"].mean() if not vol_total_mes.empty else 100

    # ── 3. Preparar datos de perfiles para muestreo ────────────────────────
    # Asumo que tu función 'normaliza_data' devuelve el df limpio
    try:
        df_norm = normaliza_data(data_original)
    except NameError:
        df_norm = data_original.copy() # Fallback por si la función no está definida en este scope

    df_norm = df_norm[df_norm["ESTATUS_VICTIMA"].isin(["DESAPARECIDA", "NO LOCALIZADA"])].copy()
    df_norm["FECHA_DESAPARICION_ANIO"] = pd.to_datetime(df_norm["FECHA_DESAPARICION"], errors="coerce").dt.year
    
    # Tomar perfiles de los últimos 2 años para muestrear
    anio_actual = pd.Timestamp.today().year
    df_reciente = df_norm[df_norm["FECHA_DESAPARICION_ANIO"] >= (anio_actual - 2)].copy()
    if df_reciente.empty:
        df_reciente = df_norm.copy()

    features_perfil_disponibles = [c for c in features_perfil if c in df_reciente.columns]
    df_reciente_limpio = df_reciente[features_perfil_disponibles].copy()

    # Limpieza estricta de numéricos para el modelo RF
    for col in df_reciente_limpio.columns:
        df_reciente_limpio[col] = pd.to_numeric(df_reciente_limpio[col], errors="coerce")
    df_reciente_limpio = df_reciente_limpio.fillna(df_reciente_limpio.median(numeric_only=True))
    cols_validas = df_reciente_limpio.select_dtypes(include=[np.number]).columns.tolist()
    df_reciente_limpio = df_reciente_limpio[cols_validas]

    # ── 4. Generar meses futuros ───────────────────────────────────────────
    meses_futuros = pd.date_range(
        start=pd.Timestamp.today().to_period("M").to_timestamp() + pd.DateOffset(months=1),
        periods=anios_forecast * 12,
        freq="MS",
    )

    # ── 5. Simulaciones Monte Carlo ────────────────────────────────────────
    resultados_sims = []
    
    # Barra de progreso para evitar que la app parezca congelada
    progress_text = "Corriendo simulaciones de Random Forest. Por favor espera..."
    my_bar = st.progress(0, text=progress_text)

    for sim in range(n_simulaciones):
        registros_mes = []
        paso_temporal = 1

        for mes in meses_futuros:
            # APLICAMOS LA TENDENCIA AL VOLUMEN: Base + (Crecimiento mensual * meses transcurridos)
            n_casos_proyectados = int(volumen_base_actual + (slope * paso_temporal))
            n_casos_proyectados = max(10, n_casos_proyectados) # Evitar muestrear números negativos o cero
            
            muestra = df_reciente_limpio.sample(
                n=n_casos_proyectados, replace=True, random_state=sim * 100 + mes.month
            ).copy()

            # Inyectar características temporales para el RF
            muestra["FECHA_DESAPARICION_ANIO"]      = mes.year
            muestra["FECHA_DESAPARICION_MES"]       = mes.month
            muestra["FECHA_DESAPARICION_DIA"]       = 15
            muestra["FECHA_DESAPARICION_DIASEMANA"] = mes.dayofweek
            muestra["FECHA_DESAPARICION_TRIMESTRE"] = mes.quarter

            # Rellenar features faltantes que el modelo exige con 0 (Cuidado con esto en RF)
            for col in features_modelo:
                if col not in muestra.columns:
                    muestra[col] = 0
            
            # Reordenar columnas para que coincidan exactamente con el entrenamiento
            muestra = muestra[features_modelo]

            # Predicción
            preds = model.predict(muestra)
            
            registros_mes.append({
                "MES"          : mes,
                "DESAPARECIDA" : (preds == "DESAPARECIDA").sum(),
                "NO LOCALIZADA": (preds == "NO LOCALIZADA").sum(),
            })
            paso_temporal += 1

        df_sim = pd.DataFrame(registros_mes)
        df_sim["SIM"] = sim
        resultados_sims.append(df_sim)
        
        # Actualizar barra de progreso
        my_bar.progress((sim + 1) / n_simulaciones, text=progress_text)
        
    my_bar.empty() # Borrar barra al terminar

    df_todas_sims = pd.concat(resultados_sims, ignore_index=True)

    # ── 6. Agregar simulaciones ────────────────────────────────────────────
    agg_forecast = (
        df_todas_sims.groupby("MES")
        .agg(
            DESAPARECIDA_MEAN  =("DESAPARECIDA",  "mean"),
            DESAPARECIDA_UPPER =("DESAPARECIDA",  "max"),
            DESAPARECIDA_LOWER =("DESAPARECIDA",  "min"),
            NOLOC_MEAN         =("NO LOCALIZADA", "mean"),
            NOLOC_UPPER        =("NO LOCALIZADA", "max"),
            NOLOC_LOWER        =("NO LOCALIZADA", "min"),
        )
        .reset_index()
    )

    # ── 7. Figura Plotly ───────────────────────────────────────────────────
    fig = go.Figure()
    fecha_corte = df_hist["FECHA_DESAPARICION"].max()

    config_clases = [
        ("DESAPARECIDA",  "DESAPARECIDA_MEAN",  "DESAPARECIDA_UPPER",  "DESAPARECIDA_LOWER"),
        ("NO LOCALIZADA", "NOLOC_MEAN",         "NOLOC_UPPER",         "NOLOC_LOWER"),
    ]

    for clase, col_mean, col_upper, col_lower in config_clases:
        color = colores[clase]

        # Histórico real
        hist_c = hist_mensual[hist_mensual["ESTATUS_VICTIMA"] == clase].sort_values("MES")
        fig.add_trace(go.Scatter(
            x=hist_c["MES"], y=hist_c["CONTEO"],
            mode="lines",
            name=f"{clase} (histórico)",
            line=dict(color=color["linea"], width=2),
        ))

        # Banda de incertidumbre
        fig.add_trace(go.Scatter(
            x=pd.concat([agg_forecast["MES"], agg_forecast["MES"].iloc[::-1]]),
            y=pd.concat([agg_forecast[col_upper], agg_forecast[col_lower].iloc[::-1]]),
            fill="toself",
            fillcolor=color["banda"],
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ))

        # Línea central del forecast
        fig.add_trace(go.Scatter(
            x=agg_forecast["MES"],
            y=agg_forecast[col_mean].round(),
            mode="lines",
            name=f"{clase} (forecast RF)",
            line=dict(color=color["linea"], width=2, dash="dot"),
        ))

    fig.add_vline(
        x=fecha_corte.timestamp() * 1000,
        line_dash="dash", line_color="gray",
        annotation_text="Hoy",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Predicción 10 años — Simulaciones Monte Carlo + RF + Tendencia",
        xaxis_title="Fecha",
        yaxis_title="Volumen Mensual",
        hovermode="x unified",
        height=550,
    )

    return fig

def matriz_confusion_sin_confidencialidad(data_raw):
    target_col = "ESTATUS_VICTIMA" 
    data_norm = normaliza_data(data_raw)

    # Filtrado de registros que sesgan el modelo
    data_norm = data_norm[data_norm[target_col] != "CONFIDENCIAL"].copy()
    
    cols_excluir = [
        target_col, "ID_VICTIMA", "ESTATUS_MAP", "SEXO_MAP", 
        "CVE_ENT", "CVE_MUN", "FECHA_NACIMIENTO_CONFIDENCIAL",
        "FECHA_DESAPARICION_CONFIDENCIAL", "FECHA_REGISTRO_CONFIDENCIAL",
    ]

    X = data_norm.drop(columns=cols_excluir, errors="ignore")
    y = data_norm[target_col]
    
    st.markdown("**Sin CONFIDENCIAL**")
    resultado_sin = auditoria_columnas(X, y)
    
    df_sin = pd.DataFrame(resultado_sin, columns=["Variable", "Correlación"])
    df_sin["Correlación"] = df_sin["Correlación"].astype(float)
    df_sin = df_sin.sort_values("Correlación", ascending=True)

    fig_sin = go.Figure(go.Bar(
        x=df_sin["Correlación"],
        y=df_sin["Variable"],
        orientation="h",
        marker=dict(
            color=df_sin["Correlación"],
            colorscale="RdYlGn_r",
            showscale=True,
            cmin=0, cmax=1,
        ),
    ))
    fig_sin.update_layout(
        title="Sin CONFIDENCIAL",
        xaxis=dict(range=[0, 1]),
        xaxis_title="Correlación absoluta",
        yaxis_title="Variable",
        height=500,
    )
    st.plotly_chart(fig_sin, use_container_width=True)

def panel_metricas(y_test, y_pred):
    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{acc * 100:.2f}%")
    col2.metric("Precision", f"{precision * 100:.2f}%")
    col3.metric("Recall",    f"{recall * 100:.2f}%")
    col4.metric("F1-Score",  f"{f1 * 100:.2f}%")
    

def dashboard_results(y_test: pd.Series, y_pred: np.ndarray, model, X_test: pd.DataFrame, data_original: pd.DataFrame = None, X_test_scaled: pd.DataFrame = None, y_prob: np.ndarray = None):

    st.title("📊 Resultados del Modelo — Estatus Víctima")

    # ── Descripción del modelo ────────────────────────────────────────────────
    st.subheader("Modelo Random Forest Classifier")  

    st.write(
        "El modelo Random Forest es un algoritmo de aprendizaje supervisado que utiliza "
        "múltiples árboles de decisión para mejorar la precisión y evitar el sobreajuste. "
        "Es especialmente útil para problemas de clasificación con datos tabulares, como "
        "el caso de predicción del estatus de víctimas."
    )
    st.write(
        "Se escogió este modelo por su capacidad para manejar grandes conjuntos de datos, "
        "su robustez frente a outliers y su habilidad para capturar relaciones no lineales "
        "entre las características y el target. Además, el Random Forest proporciona una "
        "medida de importancia de las características, lo que ayuda a identificar cuáles "
        "variables tienen mayor impacto en la predicción del estatus de las víctimas."
    )

    st.info("🎯 Target Column: ESTATUS_VICTIMA") 
    
    st.subheader("Problema con los registros Confidencial")
    st.write(
        "El modelo muestra un rendimiento excepcionalmente alto, lo que sugiere la presencia de data leakage. "
        "Es probable que una o más características estén directamente relacionadas con el target, lo que permite al modelo predecir con casi perfecta precisión durante la evaluación, pero fallaría en un entorno de producción real."
    )
    st.text("Esto se vio cuando implementamos la curva de ROC donde el AUC fue cercano a 1.0, lo cual es sospechoso en un problema tan complejo.")
    st.text("Al revisar la importancia de las características, se identificó que la columna 'CONFIDENCIAL' tenía una correlación extremadamente alta con el target 'ESTATUS_VICTIMA', lo que indica que esta variable podría estar filtrando información que no estaría disponible en un escenario real de predicción.")
    st.text("Ya que la columna 'CONFIDENCIAL' representa 49,149 registro, que es 36% del total, es un porcentaje significativo que podría estar sesgando el modelo.")
    
    col_a, col_b = st.columns(2)

    with col_a:
        conteos = data_original["ESTATUS_VICTIMA"].str.upper().str.strip().value_counts()

        fig = go.Figure(data=go.Pie(
            labels=conteos.index.tolist(),
            values=conteos.values.tolist(),
            hole=0.4,
            marker=dict(colors=["#d62728", "#1f77b4", "#ff7f0e"]),
        ))
        fig.update_layout(title="Distribución de Estatus de Víctimas")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.write(
            "Esto es importante porque si el modelo se entrena con esta columna, "
            "es probable que aprenda a predecir el estatus de las víctimas basándose "
            "en información filtrada, lo que no reflejaría su desempeño real en producción."
        )
        st.write(
            "Hay muchos registros protegidos por confidencialidad, lo que sugiere que "
            "las víctimas asociadas podrían tener características que no se pueden revelar "
            "públicamente, complicando la tarea de predicción y aumentando el riesgo de leakage."
        )
        st.write(
            "También refleja cómo el crimen organizado está creciendo, haciendo que cada vez "
            "más víctimas tengan estatus confidencial para proteger su identidad y la de sus "
            "familiares, dificultando el análisis y predicción."
        )

    st.warning(
        "Es crucial revisar las características utilizadas en el modelo para identificar "
        "posibles fuentes de leakage. Si se detecta que una variable tiene una correlación "
        "extremadamente alta con el target, se debe eliminar o modificar antes de reentrenar."
    )

    # ── Heatmap de correlación de features con el target ─────────────────────
    st.subheader("🔍 Análisis de Correlación con el Target")
    
    col1, col2 = st.columns(2)

    with col1:
        matriz_confusion_sin_confidencialidad(data_raw=data_original)

    with col2:
        st.markdown("**Con CONFIDENCIAL**")
        # Usar data_original completo
        target_col = "ESTATUS_VICTIMA"
        cols_excluir = [
            target_col,
            "ID_VICTIMA",
            "ESTATUS_MAP",
            "SEXO_MAP",
            "FECHA_NACIMIENTO_CONFIDENCIAL",
            "FECHA_DESAPARICION_CONFIDENCIAL",
            "FECHA_REGISTRO_CONFIDENCIAL",
        ]
        x_con = data_original.drop(columns=["ESTATUS_VICTIMA"] + cols_excluir, errors="ignore")
        y_con = data_original["ESTATUS_VICTIMA"]
        resultado_con = auditoria_columnas(x_con, y_con)
        df_con = pd.DataFrame(resultado_con, columns=["Variable", "Correlación"])
        df_con["Correlación"] = df_con["Correlación"].astype(float)
        df_con = df_con.sort_values("Correlación", ascending=True)

        fig_con = go.Figure(go.Bar(
            x=df_con["Correlación"],
            y=df_con["Variable"],
            orientation="h",
            marker=dict(
                color=df_con["Correlación"],
                colorscale="RdYlGn_r",
                showscale=True,
                cmin=0, cmax=1,
            ),
        ))
        fig_con.update_layout(
            title="Con CONFIDENCIAL",
            xaxis=dict(range=[0, 1]),
            xaxis_title="Correlación absoluta",
            yaxis_title="Variable",
            height=500,
        )
        st.plotly_chart(fig_con, use_container_width=True)
    st.info(
        "Por el momento, para entrenar el modelo se decidió eliminar los registros "
        "CONFIDENCIAL. Esto no es una solución definitiva, sino un paso temporal para "
        "evaluar el rendimiento del modelo sin esa variable."
    )
    
    # ── 2. Matriz de confusión ────────────────────────────────────────────────
    st.divider()
    st.title("🔍 Análisis de Resultados")
    
    panel_metricas(y_test, y_pred)
    
    st.subheader("¿Qué significan estas métricas?")
    with st.expander("Accuracy", expanded=True):
        st.write("Lo que representa accuracy es la proporción de predicciones correctas sobre el total de casos evaluados. En este contexto, un accuracy muy alto podría ser engañoso debido al desequilibrio de clases o al data leakage.")
        st.write("En este caso obtuvimos 89%, lo que sugiere que el modelo está prediciendo correctamente la mayoría de los casos. Por lo que el 20% restante representa los casos que el modelo no logró clasificar correctamente.")
        st.write("Lo indica que el modelo tiene un buen desempeño general, pero es importante analizar otras métricas para entender mejor su rendimiento, especialmente en casos de desequilibrio de clases o presencia de data leakage.")
        st.write("Formula del Accuracy:")
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
    with st.expander("Precision", expanded=True):
        st.write("La precisión en machine learning se refiere a la proporción de verdaderos positivos sobre el total de predicciones positivas realizadas por el modelo. En otras palabras, mide cuántas de las predicciones positivas del modelo son realmente correctas.")
        st.write("En este caso, una precisión del 93% indica que de todas las veces que el modelo predijo un estatus específico para una víctima, el 80.06% de esas predicciones fueron correctas. Esto sugiere que el modelo es bastante confiable al hacer predicciones positivas, aunque es importante considerar esta métrica junto con el recall para obtener una imagen completa del rendimiento del modelo.")
        st.write("Una alta precisión es especialmente valiosa en situaciones donde las predicciones falsas positivas pueden tener consecuencias significativas, como en la identificación de víctimas desaparecidas, donde un falso positivo podría llevar a esfuerzos de búsqueda innecesarios o a la preocupación de las familias.")
        st.write("En este contexto indica que las de que la predicción del estatus de las víctimas es bastante precisa.")
        st.write("Fórmula de la Precisión:")
        st.latex(r"\text{Precisión} = \frac{TP}{TP + FP}")
    with st.expander("Recall", expanded=True):
        st.write("El recall en machine es una métrica complementaria a la precisión, donde indica qué tan bueno es el modelo para encontrar o recordar todos los casos relevantes que existen en un conjunto de datos.")
        st.write("La métrica identifico 89% de los casos reales, lo que sugiere que el modelo es bastante efectivo para identificar correctamente a las víctimas desaparecidas, aunque también es importante considerar la precisión para entender el equilibrio entre falsos positivos y falsos negativos.")
        st.write("Como vemos tiene el mismo valor que el accuracy, lo que significa .")
        st.write("En el contexto de predicción de estatus de víctimas, un recall alto es crucial para asegurar que se identifiquen la mayor cantidad posible de casos reales, lo que puede ser vital para la búsqueda y asistencia a las víctimas desaparecidas.")
        st.write("Fórmula del Recall:")
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
    with st.expander("F1-Score", expanded=True):
        st.write("La es una métrica que combina la Precisión y el Recall en un solo número, dándote una calificación global del rendimiento de tu modelo. Es especialmente útil cuando tienes un desequilibrio de clases, ya que te ayuda a entender cómo el modelo está manejando tanto los falsos positivos como los falsos negativos.") 
        st.write("De acuerdo a nuestras métricas se tiene un 91%, lo que indica que el modelo tiene un buen equilibrio entre precisión y recall, aunque es importante seguir analizando otras métricas y la matriz de confusión para obtener una imagen completa del rendimiento del modelo, ya que al tener precisión como 80.06% y recall 93% el F1-Score se ve afectado por la diferencia entre ambas métricas.")
        st.write("En el contexto de predicción de estatus de víctimas, un F1-Score alto es deseable porque indica que el modelo está haciendo un buen trabajo tanto en identificar correctamente a las víctimas desaparecidas (recall) como en evitar falsos positivos (precisión), lo que es crucial para la efectividad de las acciones de búsqueda y asistencia.")
        st.write("Fórmula del F1-Score:")
        st.latex(r"F1 = 2 \times \frac{\text{Precisión} \times \text{Recall}}{\text{Precisión} + \text{Recall}}")
        st.info("**Media Harmónica vs Media Aritmética** \n Media harmónica para F1-Score:\n Se debe de usar la media harmónica en lugar de la media aritmética para el F1-Score porque esta última puede ser engañosa cuando hay un desequilibrio entre precisión y recall. La media aritmética podría dar un valor alto incluso si una de las métricas es muy baja, mientras que la media harmónica penaliza más los valores extremos, proporcionando una evaluación más equilibrada del rendimiento del modelo.\n Además La media armónica castiga fuertemente los valores extremos. Para que el F1-Score sea alto, tanto la Precisión como el Recall deben ser altos. Si uno de los dos cae a cerca de 0, el F1-Score también se desplomará, reflejando que el modelo tiene una falla grave.")
    
    texto_analisis = """
    **¿Cómo traducimos estas métricas al contexto operativo del problema?**

    *   📊 **Evaluación Rigurosa:** En el dominio crítico de la predicción de personas desaparecidas, un *accuracy* elevado puede ser un espejismo si existe desequilibrio de clases. El análisis profundo de la Precisión y el *Recall* es indispensable para auditar la robustez matemática del modelo.
    *   ⚠️ **Control de Riesgos y *Data Leakage*:** Aunque el rendimiento predictivo actual es sobresaliente, mantenemos un escrutinio meticuloso. La alta correlación detectada entre la variable `CONFIDENCIAL` y el *target* requiere investigación, ya que podría indicar una fuga de datos que infle artificialmente las métricas.
    *   ⚖️ **Impacto Estratégico:** El balance de estas métricas determina la viabilidad operativa del sistema. Maximizar el **Recall** garantiza la identificación efectiva de las víctimas, mientras que una alta **Precisión** minimiza los falsos positivos. Comprender esto es vital para dirigir estratégicamente los recursos en las operaciones de búsqueda y asistencia.
    """

    st.info(texto_analisis)

    st.divider()
    
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Matriz de Confusión")
        labels = sorted(y_test.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
        ))
        fig.update_layout(
            xaxis_title="Predicho",
            yaxis_title="Real",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("¿Qué significa?")
        st.write("La matriz de confusión muestra la distribución de las predicciones del modelo en comparación con los valores reales. Las filas representan las clases reales, mientras que las columnas representan las clases predichas. En un escenario ideal, los valores en la diagonal principal (verdaderos positivos y verdaderos negativos) serían altos, mientras que los valores fuera de la diagonal (falsos positivos y falsos negativos) serían bajos. En este caso, el modelo muestra un buen desempeño, pero es crucial analizar esta matriz junto con las métricas para entender completamente el rendimiento del modelo y detectar posibles problemas como el data leakage.")
    texto_matriz = "- 14669→ Predijo DESAPARECIDA y era DESAPARECIDA ✅ (Verdaderos Positivos) \n - 565 → Predijo NO LOCALIZADA y era NO LOCALIZADA ✅ (Verdaderos Negativos)\n - 406 → Predijo NO LOCALIZADA pero era DESAPARECIDA ❌ (Falsos Negativos )\n - 1308 → Predijo DESAPARECIDA pero era NO LOCALIZADA ❌ (Falsos Positivos) "
    st.write(texto_matriz)
    st.write("En el contexto de predicción de estatus de víctimas, un alto número de verdaderos positivos es crucial para asegurar que se identifiquen correctamente a las víctimas desaparecidas, mientras que un bajo número de falsos positivos es importante para evitar alarmas innecesarias y preocupaciones para las familias de las víctimas.")
    st.write("Sin embargo tenemos un margen de error del 20% que representa los casos que el modelo no logró clasificar correctamente, lo que sugiere que hay espacio para mejorar el modelo, especialmente en la reducción de falsos negativos, ya que es crucial identificar a la mayor cantidad posible de víctimas desaparecidas.")

    st.info("**Simulación de Umbral** se encuentra en el apartado ``simulador de umbral`` donde se puede ajustar el umbral de decisión para observar cómo afecta las métricas de precisión, recall y F1-Score, lo que es especialmente útil para encontrar el equilibrio óptimo entre estas métricas en función de las prioridades del problema.")
    st.divider()
    
    st.subheader("📈 Curva ROC sin la Columna Confidencial")
    st.write(
        "Mide la capacidad del modelo para distinguir entre clases. "
        "Cuanto más cerca del 1.0 el AUC, mejor el modelo. "
        "La línea punteada representa un clasificador aleatorio (AUC = 0.5)."
    )
    
    fig_roc = plot_roc_curve(model, X_test_scaled, y_test)
    st.plotly_chart(fig_roc, use_container_width=True)
    
    visualizar_importancia_plotly(model, X_test)
    
    st.info("***La curva de ROC***\n La curva de ROC (Receiver Operating Characteristic) es una herramienta gráfica que se utiliza para evaluar el rendimiento de un modelo de clasificación. En esta curva, el eje X representa la tasa de falsos positivos (FPR) y el eje Y representa la tasa de verdaderos positivos (TPR). Cada punto en la curva corresponde a un umbral de decisión diferente utilizado por el modelo para clasificar las instancias. Un modelo perfecto tendría un AUC (Área Bajo la Curva) de 1.0, lo que indicaría que puede distinguir perfectamente entre las clases. En este caso, al eliminar la columna CONFIDENCIAL, se espera que el AUC sea más realista y refleje mejor el desempeño del modelo sin la influencia de posibles fugas de datos.")
    st.text("Podemos ver en la curva que tenemos una identidad que sería el modelo perfecto, esto se pone para poder identificar el AUC es de 89% acertero, nos dice que el  modelo es muy robusto para distinguir entre los dos estatus de las víctimas.  Ya que tiene una buena capacidad de discriminación, lo que significa que si elegimos un caso de NO LOCALIZADA y uno de DESAPARECIDA al azar, el modelo los clasificará correctamente el 89% de las veces.  Como las curvas se alejan significativamente de la identidad, confirma que el modelo ahora sí están aportando valor predictivo real.  \n  El hecho de que la curva suba rápidamente hacia la esquina superior izquierda indica que el modelo logra una Tasa de Verdaderos Positivos (TPR) muy alta sin cometer demasiados Falsos Positivos (FPR). Como la curva se generó con el set de prueba (X_test_scaled), confirma que el modelo no memorizó los datos, sino que aprendió patrones generales.  Modelo Listo: Con un AUC de 0.89, este modelo ya tiene un nivel de confianza suficiente para ser utilizado en un entorno de análisis real. \n Además en el gráfico de abajo podemos ver las variables que influyen más en la predicción del modelo, lo que nos da una idea de qué características son más relevantes para determinar el estatus de las víctimas. Esto también puede ayudar a identificar posibles fuentes de data leakage si alguna variable tiene una importancia desproporcionadamente alta y en la clasifiación de ROC lo que causa la curva en forma de L.")
    st.text("Por otro lado tenemos las dos líneas de DESAPARECIDA y NO LOCALIZADA, que representan el desempeño del modelo para cada clase. Si estas líneas están cerca de la identidad, significa que el modelo tiene un buen desempeño para esa clase. ")
    
    st.divider()

    # ── 3. Distribución predicciones vs real ─────────────────────────────────
    st.subheader("Distribución de Predicciones vs. Valores Reales")
    df_comp  = pd.DataFrame({"Real": y_test.values, "Predicho": y_pred})
    df_real  = df_comp["Real"].value_counts().reset_index()
    df_pred  = df_comp["Predicho"].value_counts().reset_index()
    df_real.columns  = ["Clase", "Conteo"]
    df_pred.columns  = ["Clase", "Conteo"]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Real",    x=df_real["Clase"], y=df_real["Conteo"]))
    fig2.add_trace(go.Bar(name="Predicho", x=df_pred["Clase"], y=df_pred["Conteo"]))
    fig2.update_layout(barmode="group", xaxis_title="Clase", yaxis_title="Cantidad")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.text("En esta distribución podemos ver que los casos predichos con las personas DESAPARECIDAS concuerdan un .")
    st.text("Por otro lado, tenemos los casos predichos con las personas NO LOCALIZADAS; donde supera a los casos reales, por lo que sugiera que el modelo esta sobreajustando los casos de NO LOCALIZADA, lo que podría ser un indicio de data leakage o un desequilibrio en los datos. También puede ser que algunos casos de CONFIDENCIAL se esten tomando como NO LOCALIZADA, lo que explicaría el aumento en esa categoría. Esto sugiere que el modelo podría estar aprendiendo a predecir NO LOCALIZADA basándose en características que están correlacionadas con esa clase, lo que podría ser un indicio de data leakage o un desequilibrio en los datos.")

    st.divider()
    
    st.subheader("Serie de Tiempo — Casos por Año y Estatus")
    fig_ts = plot_serie_tiempo(
        X_test        = X_test,
        y_test        = y_test,
        y_pred        = y_pred,
        data_original = data_original
    )   
    st.plotly_chart(fig_ts, use_container_width=True)
    
    st.text("Se tiene que ajustar los valores predichos , ya que son valores test que representan un 20% del total, por lo que se multiplican por 5 para escalarlo al total y compararlo con los datos históricos reales.")
    st.text("Con esto podemos ver a partir de la decada de los 90s el numero de casos empezo a aumentar. siendo su pico en 2017, ")
    st.divider()
    st.subheader("🔮 Forecast con el modelo Prophet— Próximos 10 años")
    st.write(
        "Proyección basada en Prophet (Meta) entrenado con el histórico diario. "
        "La banda sombreada representa el intervalo de confianza al 95%."
    )

    fig_forecast = plot_forecast_prophet(data_original=data_original)
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.write("***¿Por qué se utilizo Prophet?***\n Prophet es un modelo de series de tiempo desarrollado por Meta que es especialmente efectivo para capturar tendencias, estacionalidades y patrones complejos en datos temporales. Se eligió Prophet para el forecast a largo plazo porque puede manejar cambios estructurales, eventos especiales y patrones no lineales que son comunes en series de tiempo relacionadas con fenómenos sociales como las desapariciones. Además, Prophet proporciona intervalos de confianza para sus predicciones, lo que es crucial para entender la incertidumbre inherente a las proyecciones a largo plazo.")

    st.info(
        "**Interpretación:** La línea punteada muestra la tendencia central estimada. "
        "El área sombreada indica el rango probable de casos — cuanto más amplia, "
        "mayor incertidumbre. Los patrones semanales y anuales son capturados automáticamente por Prophet."
    )
    
    st.divider()
    
    st.subheader("🌲 Forecast Random Forest — Próximos 10 años")
    st.write(
        "Proyección generada con el mismo modelo Random Forest entrenado. "
        "Se construyen features de fecha para cada día futuro y el modelo predice el estatus esperado. "
        "La banda representa la incertidumbre basada en la desviación estándar de las probabilidades."
    )
    fig_forecast_rf = plot_forecast_rf(model=model, data_original=data_original)
    st.plotly_chart(fig_forecast_rf, use_container_width=True)
    
    st.write("Para poder hacer un forecasting con el modelo de Random Forest, se utilizo la simulación de Monte Carlo + RF + Tedencia, debido a que el modelo de Random Forest no es adecuado para capturar tendencias a largo plazo, se implementó una simulación de Monte Carlo que genera múltiples escenarios futuros basados en el modelo Random Forest. Esta técnica permite observar la variabilidad en las predicciones y proporciona un rango de posibles resultados, lo que es especialmente útil para entender la incertidumbre inherente a las proyecciones a largo plazo.\n **RF** se basa en patrones aprendidos de los datos de entrenamiento, por lo que su capacidad para predecir tendencias futuras es limitada, y es probable que su forecast se base principalmente en patrones históricos sin considerar factores externos o cambios en el comportamiento a lo largo del tiempo. **Tendencia** se refiere a la incorporación de una tendencia lineal o no lineal en el forecast para capturar cambios a largo plazo que el modelo de RF no puede detectar por sí solo.")
    
    with st.expander("¿Por qué el forecast con Random Forest es menos confiable para tendencias a largo plazo?", expanded=True):
        st.write(
            "Random Forest es un modelo de aprendizaje supervisado que se basa en patrones aprendidos de los datos de entrenamiento. Si bien puede capturar patrones estacionales o de fecha, no está diseñado para extrapolar tendencias a largo plazo, especialmente en series de tiempo con cambios estructurales o eventos inesperados. Por lo tanto, su capacidad para predecir tendencias futuras es limitada, y es probable que su forecast se base principalmente en patrones históricos sin considerar factores externos o cambios en el comportamiento a lo largo del tiempo."
        )
    
    st.info("***Simulación de Monte Carlo con Random Forest*** \n Para abordar la incertidumbre en las predicciones a largo plazo, se implementó una simulación de Monte Carlo que genera múltiples escenarios futuros basados en el modelo Random Forest. Esta técnica permite observar la variabilidad en las predicciones y proporciona un rango de posibles resultados, lo que es especialmente útil para entender la incertidumbre inherente a las proyecciones a largo plazo.")

    st.warning(
        "⚠️ **Limitación importante:** Random Forest no fue diseñado para series de tiempo. "
        "Su forecast se basa únicamente en patrones de fecha (mes, día de semana, trimestre), "
        "por lo que no puede capturar tendencias crecientes o decrecientes a largo plazo. "
        "Para tendencias futuras, el forecast con Prophet es más confiable."
    )
    
    st.divider()

    # ── 5. Reporte de clasificación ───────────────────────────────────────────
    st.subheader("📋 Reporte de Clasificación")
    report    = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).T.round(3)
    st.dataframe(df_report, use_container_width=True)

    st.text("")
    
    st.divider()

    # ── 6. Leakage y conclusiones ─────────────────────────────────────────────
    st.subheader("⚠️ Errores comunes y posibles mejoras")  
    st.subheader("Leakage Detection")
    st.write("**¿Existe data leakage?**")
    st.write(
        "El data leakage ocurre cuando el modelo tiene acceso a información que no estaría "
        "disponible en un escenario real de predicción, lo que puede llevar a un rendimiento "
        "artificialmente alto durante la evaluación pero pobre en producción."
    )
    st.write(
        "En este análisis, se implementó una función de detección de leakage que evalúa la "
        "correlación entre cada característica y el target. Si se encuentra una correlación "
        "sospechosamente alta, se recomienda investigar esa variable."
    )
    st.warning(
        "Si se detecta leakage, se deben eliminar o modificar las características "
        "problemáticas antes de volver a entrenar el modelo."
    )
    st.subheader("Desbalance de clases")
    st.write(
        "El desbalance de clases ocurre cuando una clase tiene significativamente más ejemplos que otra, lo que puede llevar a que el modelo tenga un sesgo hacia la clase mayoritaria. En este caso, es importante analizar la distribución de las clases en el conjunto de datos y considerar técnicas como el sobremuestreo, submuestreo o el uso de métricas específicas para evaluar el rendimiento del modelo en cada clase."
    )
    st.write("Ya que en la parte del y_test tenemos que la clase DESAPARECIDA tiene 14669 casos, mientras que la clase NO LOCALIZADA tiene 565 casos, lo que indica un desbalance significativo entre las clases.")
    st.write("Esto lo podemos ver en la distribución de las clases en el conjunto de datos, donde la clase DESAPARECIDA representa aproximadamente el 94% de los casos, mientras que la clase NO LOCALIZADA representa solo el 6%. Este desbalance puede afectar el rendimiento del modelo, ya que podría aprender a predecir principalmente la clase mayoritaria (DESAPARECIDA) y tener dificultades para identificar correctamente la clase minoritaria (NO LOCALIZADA).")
    st.divider()
    st.subheader("✅ Conclusiones")
    st.write("- El modelo de Random Forest mostró un rendimiento alto en la clasificación del estatus de las víctimas, con un accuracy del 89%, una precisión del 93%, un recall del 89% y un F1-Score del 91%. Sin embargo, se identificó la presencia de data leakage debido a la alta correlación entre la variable `CONFIDENCIAL` y el target `ESTATUS_VICTIMA`, lo que infló artificialmente las métricas de rendimiento.") 
    st.write("- También como estamos trabajando mayormente con fechas como feature variables para el forecast, el modelo de Random Forest no es el más adecuado para capturar tendencias a largo plazo, por lo que se recomienda utilizar modelos específicos para series de tiempo, como Prophet, para obtener proyecciones más confiables.")
    st.write("- Gracias a este análisis, se identificaron áreas clave para mejorar el modelo, como la eliminación de características que causan data leakage y la consideración de técnicas para manejar el desbalance de clases, lo que puede llevar a un modelo más robusto y confiable para la predicción del estatus de las víctimas desaparecidas.")

def main_dashboard():
    page_layout("Forecasting")
    y_test, y_predictions, model, X_test, data_raw,X_test_scaled, y_probs = main_forecasting()
    dashboard_results(y_test, y_predictions, model, X_test=X_test, data_original=data_raw, X_test_scaled=X_test_scaled, y_prob=y_probs)
    return y_test, y_predictions, model, X_test, data_raw


main_dashboard()