import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def simulador_umbral(model, X_test, y_test, clases=None):
    """
    Simulador interactivo del umbral de decisión.
    
    Parámetros:
        model    : modelo entrenado con método predict_proba()
        X_test   : features de prueba (DataFrame o array)
        y_test   : etiquetas reales
        clases   : lista con nombres de clases, ej: ["DESAPARECIDA", "NO LOCALIZADA"]
                   Si es None se infiere del modelo.
    """

    st.write("### Simulador de Umbral de Decisión")
    st.caption(
        "El modelo calcula probabilidades internamente. El umbral define a partir "
        "de qué probabilidad se clasifica como positivo (DESAPARECIDA). "
        "Mueve el slider para ver el impacto en la matriz de confusión y las métricas."
    )

    # ── Validar que el modelo tenga predict_proba ─────────────────────────────
    if not hasattr(model, "predict_proba"):
        st.error("El modelo no tiene método predict_proba(). Solo funciona con modelos probabilísticos.")
        return

    # ── Obtener probabilidades ────────────────────────────────────────────────
    if clases is None:
        clases = list(model.classes_)

    clase_positiva = clases[0]  # DESAPARECIDA
    idx_positiva   = list(model.classes_).index(clase_positiva)
    probabilidades = model.predict_proba(X_test)[:, idx_positiva]
    y_real         = np.array(y_test)

    # ── Slider de umbral ──────────────────────────────────────────────────────
    umbral = st.slider(
        "Umbral de decisión",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        format="%.2f",
        help="Por debajo de 0.5 el modelo es más sensible (detecta más desaparecidas). "
             "Por encima es más conservador (menos falsas alarmas)."
    )

    # ── Calcular predicciones con el umbral elegido ───────────────────────────
    y_pred = np.where(probabilidades >= umbral, clase_positiva, clases[1])

    # ── Calcular matriz de confusión manualmente ──────────────────────────────
    TP = int(np.sum((y_real == clase_positiva) & (y_pred == clase_positiva)))
    FN = int(np.sum((y_real == clase_positiva) & (y_pred != clase_positiva)))
    FP = int(np.sum((y_real != clase_positiva) & (y_pred == clase_positiva)))
    TN = int(np.sum((y_real != clase_positiva) & (y_pred != clase_positiva)))

    # ── Métricas ──────────────────────────────────────────────────────────────
    precision    = TP / (TP + FP)       if (TP + FP) > 0 else 0.0
    recall       = TP / (TP + FN)       if (TP + FN) > 0 else 0.0
    f1           = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exactitud    = (TP + TN) / (TP + TN + FP + FN)
    especificidad = TN / (TN + FP)      if (TN + FP) > 0 else 0.0

    # ── Tarjetas de métricas ──────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Exactitud",       f"{exactitud:.1%}")
    c2.metric("Precisión",       f"{precision:.1%}")
    c3.metric("Recall",          f"{recall:.1%}",
              delta=f"{recall - 0.80:.1%} vs umbral 0.5" if umbral != 0.5 else None)
    c4.metric("F1-Score",        f"{f1:.1%}")
    c5.metric("Especificidad",   f"{especificidad:.1%}")

    # ── Alerta según FN ───────────────────────────────────────────────────────
    if FN > FP * 3:
        st.warning(
            f"⚠️ **FN alto ({FN:,})**: el modelo no detecta {FN:,} personas desaparecidas reales. "
            f"Considera bajar el umbral."
        )
    elif FP > TP * 0.5:
        st.warning(
            f"⚠️ **FP alto ({FP:,})**: demasiadas falsas alarmas. "
            f"Considera subir el umbral."
        )
    else:
        st.success(f"Equilibrio razonable con umbral = {umbral:.2f}")

    # ── Matriz de confusión ───────────────────────────────────────────────────

        st.write("#### Matriz de confusión")

        z      = [[TP, FN], [FP, TN]]
        texto  = [
            [f"<b>{TP:,}</b><br>TP", f"<b>{FN:,}</b><br>FN"],
            [f"<b>{FP:,}</b><br>FP", f"<b>{TN:,}</b><br>TN"],
        ]
        colores_cm = [
            ["#1B4F8A", "#E74C3C"],
            ["#F39C12", "#1B8A4F"],
        ]

        fig_cm = go.Figure(go.Heatmap(
            z=z,
            x=[f"Pred: {clases[0]}", f"Pred: {clases[1]}"],
            y=[f"Real: {clases[0]}", f"Real: {clases[1]}"],
            colorscale=[[0, "#F8F9FA"], [0.5, "#AED6F1"], [1, "#1B4F8A"]],
            showscale=False,
            hovertemplate="<b>%{text}</b><extra></extra>",
            text=texto,
            texttemplate="%{text}",
            textfont={"size": 16},
        ))

        anotaciones = [
            dict(x=0, y=0, text=f"<b>{TP:,}</b><br>TP", showarrow=False,
                 font=dict(size=16, color="white")),
            dict(x=1, y=0, text=f"<b>{FN:,}</b><br>FN", showarrow=False,
                 font=dict(size=16, color="#7B241C")),
            dict(x=0, y=1, text=f"<b>{FP:,}</b><br>FP", showarrow=False,
                 font=dict(size=16, color="#784212")),
            dict(x=1, y=1, text=f"<b>{TN:,}</b><br>TN", showarrow=False,
                 font=dict(size=16, color="#1D8348")),
        ]

        fig_cm.update_layout(
            annotations=anotaciones,
            xaxis_title="Predicho",
            yaxis_title="Real",
            height=320,
            margin=dict(t=20, b=40, l=80, r=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Curva Precisión-Recall vs Umbral ──────────────────────────────────────
        st.write("#### Métricas por umbral")

        umbrales  = np.linspace(0.01, 0.99, 100)
        precs, recs, f1s = [], [], []

        for u in umbrales:
            yp  = np.where(probabilidades >= u, clase_positiva, clases[1])
            tp_ = int(np.sum((y_real == clase_positiva) & (yp == clase_positiva)))
            fn_ = int(np.sum((y_real == clase_positiva) & (yp != clase_positiva)))
            fp_ = int(np.sum((y_real != clase_positiva) & (yp == clase_positiva)))
            tn_ = int(np.sum((y_real != clase_positiva) & (yp != clase_positiva)))
            p   = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0
            r   = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
            f   = 2*p*r/(p+r) if (p+r) > 0 else 0
            precs.append(p); recs.append(r); f1s.append(f)

        fig_curva = go.Figure()
        fig_curva.add_trace(go.Scatter(
            x=umbrales, y=precs, name="Precisión",
            line=dict(color="#1B4F8A", width=2)))
        fig_curva.add_trace(go.Scatter(
            x=umbrales, y=recs, name="Recall",
            line=dict(color="#E24B4A", width=2)))
        fig_curva.add_trace(go.Scatter(
            x=umbrales, y=f1s, name="F1",
            line=dict(color="#639922", width=2)))

        # Línea vertical en el umbral actual
        fig_curva.add_vline(
            x=umbral, line_width=1.5,
            line_dash="dash", line_color="#888",
            annotation_text=f"umbral={umbral:.2f}",
            annotation_position="top right",
            annotation_font_size=11,
        )

        fig_curva.update_layout(
            height=320,
            margin=dict(t=20, b=40, l=40, r=20),
            xaxis_title="Umbral",
            yaxis_title="Valor",
            yaxis=dict(range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_curva, use_container_width=True)

    # ── Tabla resumen ─────────────────────────────────────────────────────────
        filas = []
        for u in np.arange(0.1, 1.0, 0.1):
            yp  = np.where(probabilidades >= u, clase_positiva, clases[1])
            tp_ = int(np.sum((y_real == clase_positiva) & (yp == clase_positiva)))
            fn_ = int(np.sum((y_real == clase_positiva) & (yp != clase_positiva)))
            fp_ = int(np.sum((y_real != clase_positiva) & (yp == clase_positiva)))
            tn_ = int(np.sum((y_real != clase_positiva) & (yp != clase_positiva)))
            p   = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0
            r   = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
            f   = 2*p*r/(p+r) if (p+r) > 0 else 0
            acc = (tp_+tn_)/(tp_+tn_+fp_+fn_)
            filas.append({
                "Umbral": f"{u:.1f}",
                "TP": f"{tp_:,}", "FN": f"{fn_:,}",
                "FP": f"{fp_:,}", "TN": f"{tn_:,}",
                "Precisión": f"{p:.1%}",
                "Recall":    f"{r:.1%}",
                "F1":        f"{f:.1%}",
                "Exactitud": f"{acc:.1%}",
            })
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)