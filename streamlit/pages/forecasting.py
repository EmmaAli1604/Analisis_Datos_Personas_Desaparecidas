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


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.tree import  _tree

import plotly.express as px


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

def plot_serie_tiempo(X_test, y_test, y_pred, data_original):
    clases  = ["DESAPARECIDA", "NO LOCALIZADA"]
    colores = {"DESAPARECIDA": "#1f77b4", "NO LOCALIZADA": "#ff7f0e", "CONFIDENCIAL": "#d62728"}

    # ── Datos históricos completos ────────────────────────────────────────────
    df_hist = data_original[["FECHA_DESAPARICION", "ESTATUS_VICTIMA"]].copy()
    df_hist["FECHA_DESAPARICION"] = pd.to_datetime(df_hist["FECHA_DESAPARICION"], errors="coerce")
    df_hist = df_hist.dropna(subset=["FECHA_DESAPARICION"])
    df_hist["ANIO"] = df_hist["FECHA_DESAPARICION"].dt.year
    hist_grouped = df_hist.groupby(["ANIO", "ESTATUS_VICTIMA"]).size().reset_index(name="CONTEO")

    # ── Datos predichos — escalar al 100% (multiplicar por 1/test_size) ───────
    col_anio = "FECHA_DESAPARICION_ANIO"
    if col_anio not in X_test.columns:
        st.warning(f"No se encontró '{col_anio}' en X_test")
        return go.Figure()

    factor_escala = 1 / 0.2  # ← test_size=0.2, entonces x5 para comparar con el total

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
    pred_grouped["CONTEO"] = (pred_grouped["CONTEO"] * factor_escala).round()  # ← escalar

    # ── Figura ────────────────────────────────────────────────────────────────
    fig = go.Figure()

    for clase in ["CONFIDENCIAL"] + clases:
        color = colores.get(clase, "#999999")

        # Real
        df_c = hist_grouped[hist_grouped["ESTATUS_VICTIMA"] == clase].sort_values("ANIO")
        if not df_c.empty:
            fig.add_trace(go.Scatter(
                x=df_c["ANIO"], y=df_c["CONTEO"],
                mode="lines+markers",
                name=f"{clase} (real)",
                line=dict(color=color, width=2, dash="solid"),
                marker=dict(size=5),
            ))

        # Predicho (sin CONFIDENCIAL)
        if clase == "CONFIDENCIAL":
            continue
        df_p = pred_grouped[pred_grouped["ESTATUS_PRED"] == clase].sort_values("ANIO")
        if not df_p.empty:
            fig.add_trace(go.Scatter(
                x=df_p["ANIO"], y=df_p["CONTEO"],
                mode="lines+markers",
                name=f"{clase} (predicho)",
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=5, symbol="diamond"),
            ))

    anio_min = max(1990, int(df_hist["ANIO"].min()))
    anio_max = int(df_hist["ANIO"].max()) + 1

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
        st.write("En este caso obtuvimos 80.06%, lo que sugiere que el modelo está prediciendo correctamente la mayoría de los casos. Por lo que el 20% restante representa los casos que el modelo no logró clasificar correctamente.")
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
        st.write("La métrica identifico 80.06% de los casos reales, lo que sugiere que el modelo es bastante efectivo para identificar correctamente a las víctimas desaparecidas, aunque también es importante considerar la precisión para entender el equilibrio entre falsos positivos y falsos negativos.")
        st.write("Como vemos tiene el mismo valor que el accuracy, lo que significa .")
        st.write("En el contexto de predicción de estatus de víctimas, un recall alto es crucial para asegurar que se identifiquen la mayor cantidad posible de casos reales, lo que puede ser vital para la búsqueda y asistencia a las víctimas desaparecidas.")
        st.write("Fórmula del Recall:")
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
    with st.expander("F1-Score", expanded=True):
        st.write("La es una métrica que combina la Precisión y el Recall en un solo número, dándote una calificación global del rendimiento de tu modelo. Es especialmente útil cuando tienes un desequilibrio de clases, ya que te ayuda a entender cómo el modelo está manejando tanto los falsos positivos como los falsos negativos.") 
        st.write("De acuerdo a nuestras métricas se tiene un 86.41%, lo que indica que el modelo tiene un buen equilibrio entre precisión y recall, aunque es importante seguir analizando otras métricas y la matriz de confusión para obtener una imagen completa del rendimiento del modelo, ya que al tener precisión como 80.06% y recall 93% el F1-Score se ve afectado por la diferencia entre ambas métricas.")
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
    texto_matriz = "- → Predijo DESAPARECIDA y era DESAPARECIDA ✅ (Verdaderos Positivos) \n - 759 → Predijo NO LOCALIZADA y era NO LOCALIZADA ✅ (Verdaderos Negativos)\n - 3,167 → Predijo NO LOCALIZADA pero era DESAPARECIDA ❌ (Falsos Negativos )\n - 212 → Predijo DESAPARECIDA pero era NO LOCALIZADA ❌ (Falsos Positivos) "
    st.write(texto_matriz)
    st.write("En el contexto de predicción de estatus de víctimas, un alto número de verdaderos positivos es crucial para asegurar que se identifiquen correctamente a las víctimas desaparecidas, mientras que un bajo número de falsos positivos es importante para evitar alarmas innecesarias y preocupaciones para las familias de las víctimas.")
    st.write("Sin embargo tenemos un margen de error del 20% que representa los casos que el modelo no logró clasificar correctamente, lo que sugiere que hay espacio para mejorar el modelo, especialmente en la reducción de falsos negativos, ya que es crucial identificar a la mayor cantidad posible de víctimas desaparecidas.")

    st.info("** Simulación de Umbral ** se encuentra en el apartado ``simulador de umbral`` donde se puede ajustar el umbral de decisión para observar cómo afecta las métricas de precisión, recall y F1-Score, lo que es especialmente útil para encontrar el equilibrio óptimo entre estas métricas en función de las prioridades del problema.")
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
    st.text("Podemos ver en la curva que tenemos una identidad que sería el modelo perfecto, esto se pone para poder identificar ")
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
    st.text("Con esto podemos ver")
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
    
    st.text("")

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
    st.write("Ya que en la parte del y_test tenemos que la clase DESAPARECIDA tiene 15977 casos, mientras que la clase NO LOCALIZADA tiene 971 casos, lo que indica un desbalance significativo entre las clases.")
    st.write("Esto lo podemos ver en la distribución de las clases en el conjunto de datos, donde la clase DESAPARECIDA representa aproximadamente el 94% de los casos, mientras que la clase NO LOCALIZADA representa solo el 6%. Este desbalance puede afectar el rendimiento del modelo, ya que podría aprender a predecir principalmente la clase mayoritaria (DESAPARECIDA) y tener dificultades para identificar correctamente la clase minoritaria (NO LOCALIZADA).")
    st.divider()
    st.subheader("✅ Conclusiones")
    st.text("") 


def main_dashboard():
    page_layout("Forecasting")
    y_test, y_predictions, model, X_test, data_raw,X_test_scaled, y_probs = main_forecasting()
    dashboard_results(y_test, y_predictions, model, X_test=X_test, data_original=data_raw, X_test_scaled=X_test_scaled, y_prob=y_probs)
    return y_test, y_predictions, model, X_test, data_raw


main_dashboard()