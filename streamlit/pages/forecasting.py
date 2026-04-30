from data.config.config import DATA_INPUT
from src.models.forecasting import data_scale, balance_data, train_model, forecasting, reporte_resultados
from sklearn.model_selection import train_test_split
import streamlit as st
from components.layout import page_layout
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    classification_report,
)

def dashboard_results(y_test: pd.Series, y_pred: np.ndarray, model, X_test: pd.DataFrame):

    st.title("📊 Resultados del Modelo — Estatus Víctima")

    # ── 1. Métricas principales ───────────────────────────────────────────────
    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-9)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{acc:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall",    f"{recall:.2%}")
    col4.metric("F1-Score",  f"{f1:.2%}")

    st.divider()

    # ── 2. Matriz de confusión ────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Matriz de Confusión")
        cm     = confusion_matrix(y_test, y_pred)
        labels = sorted(y_test.unique())

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax
        )
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        st.pyplot(fig)
        plt.close(fig)

    # ── 3. Distribución predicciones vs real ─────────────────────────────────
    with col_b:
        st.subheader("Predicho vs Real")
        df_comp = pd.DataFrame({"Real": y_test.values, "Predicho": y_pred})
        counts  = df_comp.apply(pd.Series.value_counts).fillna(0)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        counts.T.plot(kind="bar", ax=ax2, colormap="Set2", edgecolor="white")
        ax2.set_xlabel("Clase")
        ax2.set_ylabel("Cantidad")
        ax2.legend(title="Origen")
        ax2.tick_params(axis="x", rotation=0)
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()

    # ── 4. Importancia de variables ───────────────────────────────────────────
    st.subheader("🌲 Importancia de Variables (Top 15)")
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    top15       = importances.nlargest(15).sort_values()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    top15.plot(kind="barh", ax=ax3, color="steelblue", edgecolor="white")
    ax3.set_xlabel("Importancia")
    ax3.set_title("Feature Importance — Random Forest")
    st.pyplot(fig3)
    plt.close(fig3)

    st.divider()

    # ── 5. Reporte de clasificación ───────────────────────────────────────────
    st.subheader("📋 Reporte de Clasificación")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).T.round(3)
    st.dataframe(df_report, use_container_width=True)

def main():
    page_layout("Forecasting")
    
    data_imputed = pd.read_csv(DATA_INPUT)
    target_col   = "ESTATUS_VICTIMA"

    X = data_imputed.drop(columns=[target_col])
    y = data_imputed[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_bal, y_train_bal = balance_data(X_train, y_train)
    X_train_scaled = data_scale(X_train_bal)
    X_test_scaled = data_scale(X_test)

    model = train_model(X_train_scaled, y_train_bal)
    y_predictions = forecasting(model, X_test_scaled)
    
    dashboard_results(y_test, y_predictions, model, X_test_scaled)

main()