import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import confusion_matrix
import streamlit as st

def matriz_confusion_con_etiquetas(y_test, y_predictions):
    """
    Grafica la matriz de confusión con etiquetas TP, TN, FP, FN
    y métricas derivadas.
    """
    # ── Calcular matriz ───────────────────────────────────────────────────────
    clases = sorted(list(set(y_test)))  # ["DESAPARECIDA", "NO LOCALIZADA"]
    cm = confusion_matrix(y_test, y_predictions, labels=clases)

    # Asumiendo clase positiva = clases[0] (DESAPARECIDA)
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]

    # ── Etiquetas con valor + tipo ────────────────────────────────────────────
    etiquetas = [
        [f"<b>{TP}</b><br>TP<br>(Verdadero Positivo)", f"<b>{FN}</b><br>FN<br>(Falso Negativo)"],
        [f"<b>{FP}</b><br>FP<br>(Falso Positivo)",     f"<b>{TN}</b><br>TN<br>(Verdadero Negativo)"],
    ]

    colores = [
        ["#1B4F8A", "#E74C3C"],   # TP=azul oscuro (bien), FN=rojo (error grave)
        ["#F39C12", "#1B8A4F"],   # FP=naranja (error), TN=verde (bien)
    ]

    # ── Figura con subplots de texto coloreado ────────────────────────────────
    fig = go.Figure()

    # Heatmap base
    fig.add_trace(go.Heatmap(
        z=[[TP, FN], [FP, TN]],
        x=[f"Pred: {clases[0]}", f"Pred: {clases[1]}"],
        y=[f"Real: {clases[0]}", f"Real: {clases[1]}"],
        colorscale=[
            [0.0, "#F8F9FA"],
            [0.5, "#AED6F1"],
            [1.0, "#1B4F8A"],
        ],
        showscale=False,
        text=etiquetas,
        texttemplate="%{text}",
        textfont={"size": 14},
        hovertemplate="<b>%{text}</b><br>Valor: %{z}<extra></extra>",
    ))


    fig.update_layout(
        title="Matriz de Confusión — TP / TN / FP / FN",
        xaxis_title="Predicho",
        yaxis_title="Real",
        height=420,
        margin=dict(t=50, b=50, l=80, r=20),
        plot_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Advertencia si FN es alto ─────────────────────────────────────────────
    if FN > FP * 5:
        st.warning(
            f"⚠️ **Alto número de Falsos Negativos ({FN:,})**: el modelo no está identificando "
            f"correctamente a personas desaparecidas. Considera balancear el dataset "
            f"(SMOTE, class_weight='balanced') o ajustar el umbral de decisión."
        )