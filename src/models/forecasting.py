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

from .normalization import  preprocess_features, normaliza_data
from ..data.config.config import DATA_INPUT
from sklearn.preprocessing import StandardScaler


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,          # ← limita profundidad del árbol
        min_samples_leaf=10,  # ← cada hoja necesita mínimo 10 muestras
        min_samples_split=20, # ← para dividir un nodo se necesitan 20 muestras
        max_features="sqrt",  # ← usa raíz cuadrada de features por split
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
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

def detectar_leakage(X: pd.DataFrame, y: pd.Series, threshold=0.95):
    """Detecta columnas con correlación sospechosamente alta con el target."""
    from sklearn.preprocessing import LabelEncoder
    
    y_enc = LabelEncoder().fit_transform(y)
    problemas = []
    
    for col in X.columns:
        try:
            corr = abs(np.corrcoef(X[col].fillna(0), y_enc)[0, 1])
            if corr > threshold:
                problemas.append((col, round(corr, 4)))
        except Exception:
            pass
    
    if problemas:
        print("⚠️  POSIBLE DATA LEAKAGE detectado:")
        for col, corr in sorted(problemas, key=lambda x: -x[1]):
            print(f"   {col:40s} correlación={corr}")
    else:
        print("✅ No se detectó leakage obvio.")
    
    return [col for col, _ in problemas]


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

    scaler = StandardScaler()
    num_cols = x_train.select_dtypes(include="number").columns
    x_scaled = x_train.copy()
    x_scaled[num_cols] = scaler.fit_transform(x_train[num_cols])
    return x_scaled

def diagnostico(X_train, X_test, y_train, y_test, model):
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    
    print("=== DIAGNÓSTICO DE OVERFITTING ===")
    print(f"Accuracy TRAIN : {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Accuracy TEST  : {accuracy_score(y_test,  y_pred_test):.4f}")
    print(f"\nDiferencia     : {accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test):.4f}")
    print(f"\nClases en train: {dict(y_train.value_counts())}")
    print(f"Clases en test : {dict(y_test.value_counts())}")

def auditoria_columnas(X: pd.DataFrame, y: pd.Series):
    """Encuentra columnas que predigan el target perfectamente."""
    from sklearn.preprocessing import LabelEncoder
    
    y_enc = LabelEncoder().fit_transform(y)
    print("=== AUDITORÍA DE COLUMNAS ===")
    print(f"Shape del DataFrame: {X.shape}")
    print(f"\nColumnas y tipos:\n{X.dtypes.to_string()}")
    
    print("\n--- Correlación con target ---")
    for col in X.columns:
        try:
            col_enc = LabelEncoder().fit_transform(X[col].fillna("NA").astype(str))
            corr = abs(np.corrcoef(col_enc, y_enc)[0, 1])
            flag = " ⚠️  LEAKAGE" if corr > 0.7 else ""
            print(f"  {col:45s} corr={corr:.3f}{flag}")
        except Exception as e:
            print(f"  {col:45s} ERROR: {e}")

def main() -> None:
    data_raw   = pd.read_csv(DATA_INPUT)
    target_col = "ESTATUS_VICTIMA"

    # 1. Normalizar (fechas, mayúsculas, mapeos) — antes de cualquier split
    data_norm = normaliza_data(data_raw)

    # 2. Separar X e y — excluir columnas con leakage conocido
    cols_excluir = [
        target_col,
        "ESTATUS_MAP",
        "SEXO",
        "FECHA_NACIMIENTO_CONFIDENCIAL",
        "FECHA_DESAPARICION_CONFIDENCIAL",
        "FECHA_REGISTRO_CONFIDENCIAL",
        "FECHA_NACIMIENTO",
        "FECHA_DESAPARICION",
        "FECHA_REGISTRO",
    ]
    X = data_norm.drop(columns=cols_excluir, errors="ignore")
    y = data_norm[target_col]

    # 3. Auditar leakage antes de entrenar
    auditoria_columnas(X, y)

    # 4. Split estratificado ANTES de cualquier transformación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Preprocesar train y test por separado (evita data leakage del encoder)
    X_train = preprocess_features(X_train)
    X_test  = preprocess_features(X_test)

    # 6. Balancear SOLO train
    X_train_bal, y_train_bal = balance_data(X_train, y_train)

    # 7. Escalar
    X_train_scaled = data_scale(X_train_bal)
    X_test_scaled  = data_scale(X_test)

    # 8. Entrenar
    model = train_model(X_train_scaled, y_train_bal)

    # 9. Diagnóstico y evaluación
    diagnostico(X_train_scaled, X_test_scaled, y_train_bal, y_test, model)
    y_predictions = forecasting(model, X_test_scaled)
    reporte_resultados(y_test, y_predictions)


if __name__ == "__main__":
    main()