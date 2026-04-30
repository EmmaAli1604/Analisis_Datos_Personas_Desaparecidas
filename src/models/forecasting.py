import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from .data.config.config import DATA_INPUT
from sklearn.preprocessing import StandardScaler


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Entrena un RandomForestClassifier y devuelve el modelo ajustado."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    return rf


def forecasting(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Genera predicciones usando el modelo ya entrenado.

    Args:
        model:  Modelo RandomForest previamente entrenado.
        X_test: DataFrame con las características de prueba.

    Returns:
        Array con las predicciones para cada fila de X_test.
    """
    return model.predict(X_test)


def reporte_resultados(y_true: pd.Series, y_pred: np.ndarray) -> None:
    """Imprime métricas de evaluación del modelo."""
    acc       = accuracy_score(y_true, y_pred)
    cm        = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Confusion Matrix:\n{cm}")


def balance_data(
    x_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Balancea clases mediante sobremuestreo de la clase minoritaria (resample).
    Requiere solo scikit-learn, sin dependencias extra.
    """
    from sklearn.utils import resample

    df = x_train.copy()
    df["__target__"] = y_train.values

    majority_class = y_train.value_counts().idxmax()
    majority_size  = y_train.value_counts().max()

    groups = []
    for label, group in df.groupby("__target__"):
        if label != majority_class:
            group = resample(group, replace=True, n_samples=majority_size, random_state=42)
        groups.append(group)

    df_balanced = pd.concat(groups).sample(frac=1, random_state=42).reset_index(drop=True)
    y_balanced  = df_balanced.pop("__target__")
    return df_balanced, y_balanced


def data_scale(x_train: pd.DataFrame) -> pd.DataFrame:
    """Escala columnas numéricas con StandardScaler."""

    scaler      = StandardScaler()
    num_cols    = x_train.select_dtypes(include="number").columns
    x_scaled    = x_train.copy()
    x_scaled[num_cols] = scaler.fit_transform(x_train[num_cols])
    return x_scaled


def main() -> None:
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

    reporte_resultados(y_test, y_predictions)


if __name__ == "__main__":
    main()