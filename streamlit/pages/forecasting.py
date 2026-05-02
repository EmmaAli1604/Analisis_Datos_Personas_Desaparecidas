import streamlit as st
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from src.models.forecasting import main_forecasting
from src.models.forecasting import auditoria_columnas
from src.data.config.config import DATA_INPUT
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
)


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.tree import  _tree


def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series):

    classes = sorted(y_test.unique())
    y_prob  = model.predict_proba(X_test)
    y_bin   = label_binarize(y_test, classes=classes)

    # ── General: funciona para 2, 3 o N clases ────────────────────────────────
    if y_bin.shape[1] != len(classes):
        y_bin = np.hstack([1 - y_bin, y_bin])

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    fig    = go.Figure()

    for i, clase in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{clase} (AUC = {roc_auc:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Aleatorio (AUC = 0.5)",
        line=dict(color="gray", width=1, dash="dash"),
    ))

    fig.update_layout(
        title=f"Curva ROC — One vs Rest ({len(classes)} clases)",
        xaxis_title="Tasa de Falsos Positivos (FPR)",
        yaxis_title="Tasa de Verdaderos Positivos (TPR)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        legend=dict(x=0.6, y=0.1),
    )

    return fig

def plot_decision_tree(model, X_test: pd.DataFrame, tree_index: int = 0, max_depth: int = 3):
    """
    Grafica un árbol de decisión individual del Random Forest usando Plotly.
    
    Args:
        model:       RandomForestClassifier ya entrenado
        X_test:      DataFrame con las features (para nombres de columnas)
        tree_index:  Índice del árbol a visualizar (0 = primero)
        max_depth:   Profundidad máxima a mostrar (recomendado: 3-4)
    """
    tree      = model.estimators_[tree_index]
    feature_names = X_test.columns.tolist()
    classes   = [str(c) for c in model.classes_]

    # ── Extraer nodos del árbol ───────────────────────────────────────────────
    tree_ = tree.tree_
    nodes_x, nodes_y, node_text, node_color = [], [], [], []
    edge_x,  edge_y  = [], []

    def get_color(node_id):
        """Color según clase mayoritaria en el nodo."""
        values  = tree_.value[node_id][0]
        clase   = np.argmax(values)
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        return palette[clase % len(palette)]

    def traverse(node_id, x, y, dx, depth):
        if depth > max_depth:
            return

        # Texto del nodo
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            feat      = feature_names[tree_.feature[node_id]]
            threshold = tree_.threshold[node_id]
            label     = f"<b>{feat}</b><br>≤ {threshold:.2f}"
        else:
            values    = tree_.value[node_id][0]
            clase_idx = np.argmax(values)
            label     = f"<b>🍃 {classes[clase_idx]}</b><br>n={int(sum(values))}"

        nodes_x.append(x)
        nodes_y.append(y)
        node_text.append(label)
        node_color.append(get_color(node_id))

        # Hijo izquierdo
        left = tree_.children_left[node_id]
        if left != _tree.TREE_LEAF and depth < max_depth:
            child_x = x - dx
            child_y = y - 1
            edge_x.extend([x, child_x, None])
            edge_y.extend([y, child_y, None])
            traverse(left, child_x, child_y, dx / 2, depth + 1)

        # Hijo derecho
        right = tree_.children_right[node_id]
        if right != _tree.TREE_LEAF and depth < max_depth:
            child_x = x + dx
            child_y = y - 1
            edge_x.extend([x, child_x, None])
            edge_y.extend([y, child_y, None])
            traverse(right, child_x, child_y, dx / 2, depth + 1)

    traverse(0, x=0, y=0, dx=2 ** (max_depth - 1), depth=0)

    # ── Construir figura ──────────────────────────────────────────────────────
    fig = go.Figure()

    # Aristas
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#cccccc", width=1),
        hoverinfo="none",
    ))

    # Nodos
    fig.add_trace(go.Scatter(
        x=nodes_x, y=nodes_y,
        mode="markers+text",
        marker=dict(size=50, color=node_color, line=dict(color="white", width=2)),  # ← size 30 → 50
        text=node_text,
        textposition="middle center",
        hoverinfo="text",
        textfont=dict(size=9, color="black"),  # ← "white" → "black"
    ))

    fig.update_layout(
        title=f"Árbol #{tree_index} del Random Forest (profundidad máx. {max_depth})",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig

def plot_serie_tiempo(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    fecha_col: str = "FECHA_DESAPARICION",
    data_original: pd.DataFrame = None, 
):
    """
    Serie de tiempo de casos por año distinguiendo datos reales vs predichos.
    Requiere que data_original tenga FECHA_DESAPARICION y ESTATUS_VICTIMA.
    """

    clases  = ["DESAPARECIDA", "NO LOCALIZADA", "CONFIDENCIAL"]
    colores = {"DESAPARECIDA": "#1f77b4", "NO LOCALIZADA": "#ff7f0e", "CONFIDENCIAL": "#d62728"}

    # ── Datos históricos completos (reales) ───────────────────────────────────
    df_hist = data_original[["FECHA_DESAPARICION", "ESTATUS_VICTIMA"]].copy()
    df_hist["FECHA_DESAPARICION"] = pd.to_datetime(df_hist["FECHA_DESAPARICION"], errors="coerce")
    df_hist = df_hist.dropna(subset=["FECHA_DESAPARICION"])
    df_hist["ANIO"] = df_hist["FECHA_DESAPARICION"].dt.year
    hist_grouped = df_hist.groupby(["ANIO", "ESTATUS_VICTIMA"]).size().reset_index(name="CONTEO")

    # ── Datos predichos (solo X_test con sus fechas) ──────────────────────────
    df_pred = data_original.loc[X_test.index, "FECHA_DESAPARICION"].copy().to_frame()
    df_pred["FECHA_DESAPARICION"] = pd.to_datetime(df_pred["FECHA_DESAPARICION"], errors="coerce")
    df_pred["ESTATUS_PRED"] = y_pred
    df_pred = df_pred.dropna(subset=["FECHA_DESAPARICION"])
    df_pred["ANIO"] = df_pred["FECHA_DESAPARICION"].dt.year
    pred_grouped = df_pred.groupby(["ANIO", "ESTATUS_PRED"]).size().reset_index(name="CONTEO")

    # ── Construir figura ──────────────────────────────────────────────────────
    fig = go.Figure()

    for clase in clases:
        color = colores.get(clase, "#999999")

        # Línea sólida — datos reales
        df_c = hist_grouped[hist_grouped["ESTATUS_VICTIMA"] == clase].sort_values("ANIO")
        if not df_c.empty:
            fig.add_trace(go.Scatter(
                x=df_c["ANIO"],
                y=df_c["CONTEO"],
                mode="lines+markers",
                name=f"{clase} (real)",
                line=dict(color=color, width=2, dash="solid"),
                marker=dict(size=6),
            ))

        # Línea punteada — predicciones del modelo
        df_p = pred_grouped[pred_grouped["ESTATUS_PRED"] == clase].sort_values("ANIO")
        if not df_p.empty:
            fig.add_trace(go.Scatter(
                x=df_p["ANIO"],
                y=df_p["CONTEO"],
                mode="lines+markers",
                name=f"{clase} (predicho)",
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=6, symbol="diamond"),
            ))

    fig.update_layout(
        title="Serie de Tiempo — Casos por Año y Estatus",
        xaxis_title="Año",
        yaxis_title="Número de Casos",
        legend_title="Estatus",
        hovermode="x unified",
        xaxis=dict(dtick=1),
    )

    return fig

def dashboard_results(y_test: pd.Series, y_pred: np.ndarray, model, X_test: pd.DataFrame, data_original: pd.DataFrame = None):

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
        st.markdown("**Sin CONFIDENCIAL**")
        resultado_sin = auditoria_columnas(X_test, y_test)
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

    with col2:
        st.markdown("**Con CONFIDENCIAL**")
        # Usar data_original completo
        target_col = "ESTATUS_VICTIMA"
        cols_excluir = [
            target_col,
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
    
    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{acc * 100:.2f}%")
    col2.metric("Precision", f"{precision * 100:.2f}%")
    col3.metric("Recall",    f"{recall * 100:.2f}%")
    col4.metric("F1-Score",  f"{f1 * 100:.2f}%")
    
    st.text("")
    

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
        st.write(""
        )

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
    
    st.text("")

    st.divider()
    
    st.subheader("Serie de Tiempo — Casos por Año y Estatus")
    fig_ts = plot_serie_tiempo(
        X_test        = X_test,
        y_test        = y_test,
        y_pred        = y_pred,
        data_original = pd.read_csv(DATA_INPUT)
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    
    st.divider()

    # ── 4. Árbol de Decisión ────────────────────────────────────────────────
    st.subheader("🌲Árbol de Decisión")
    st.write(
        "Se visualiza uno de los árboles individuales que componen el Random Forest. "
        "Cada nodo muestra la variable de división y el umbral; "
        "las hojas muestran la clase predicha."
    )

    col_ctrl1, col_ctrl2 = st.columns(2)
    tree_index = col_ctrl1.slider("Árbol a visualizar", 0, len(model.estimators_) - 1, 0)
    max_depth  = col_ctrl2.slider("Profundidad máxima", 1, 5, 3)

    fig_tree = plot_decision_tree(model, X_test, tree_index=tree_index, max_depth=max_depth)
    st.plotly_chart(fig_tree, use_container_width=True)

    st.divider()
    
    # ── 5 . Curva ROC ───────────────────────────────────────────────────────────
    
    st.subheader("📈 Curva ROC sin la Columna Confidencial")
    st.write(
        "Mide la capacidad del modelo para distinguir entre clases. "
        "Cuanto más cerca del 1.0 el AUC, mejor el modelo. "
        "La línea punteada representa un clasificador aleatorio (AUC = 0.5)."
    )
    fig_roc = plot_roc_curve(model, X_test, y_test)
    st.plotly_chart(fig_roc, use_container_width=True)
    
    st.divider()

    # ── 5. Reporte de clasificación ───────────────────────────────────────────
    st.subheader("📋 Reporte de Clasificación")
    report    = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).T.round(3)
    st.dataframe(df_report, use_container_width=True)

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

    st.divider()
    st.subheader("✅ Conclusiones")
    st.text("") 


def main():
    page_layout("Forecasting")
    y_test, y_pred, model, X_test = main_forecasting()
    dashboard_results(y_test, y_pred, model, X_test, data_original=pd.read_csv(DATA_INPUT))


main()