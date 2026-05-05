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
from .normalization import normaliza_data, preprocess_features
from sklearn.metrics import roc_auc_score
from ..data.config.config import DATA_INPUT
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE 
from sklearn.impute import SimpleImputer

def investigar_confidencial(df: pd.DataFrame, target_col: str = "ESTATUS_VICTIMA"):
    """
    Compara registros CONFIDENCIAL vs el resto para encontrar
    qué los hace estructuralmente distintos.
    """
    df_conf  = df[df[target_col] == "CONFIDENCIAL"]
    df_otros = df[df[target_col] != "CONFIDENCIAL"]

    print("=" * 60)
    print(f"REGISTROS CONFIDENCIAL : {len(df_conf):,}")
    print(f"REGISTROS OTROS        : {len(df_otros):,}")
    print("=" * 60)

    alertas = []

    for col in df.columns:
        if col == target_col:
            continue

        # ── A. Proporción de NaN ──────────────────────────────────────────
        nan_conf  = df_conf[col].isna().mean()
        nan_otros = df_otros[col].isna().mean()
        diff_nan  = abs(nan_conf - nan_otros)

        # ── B. Proporción de valor "CONFIDENCIAL" como string ─────────────
        if df[col].dtype == object:
            str_conf  = df_conf[col].astype(str).str.upper().str.contains("CONFIDENCIAL").mean()
            str_otros = df_otros[col].astype(str).str.upper().str.contains("CONFIDENCIAL").mean()
            diff_str  = abs(str_conf - str_otros)
        else:
            diff_str  = 0

        # ── C. Valores únicos exclusivos de CONFIDENCIAL ──────────────────
        vals_conf  = set(df_conf[col].dropna().astype(str).unique())
        vals_otros = set(df_otros[col].dropna().astype(str).unique())
        exclusivos = vals_conf - vals_otros

        # ── Reportar si hay diferencia significativa ──────────────────────
        if diff_nan > 0.3 or diff_str > 0.3:
            nivel = "🔴 ALTO" if (diff_nan > 0.7 or diff_str > 0.7) else "🟡 MEDIO"
            alertas.append({
                "columna"        : col,
                "nivel"          : nivel,
                "nan_confidencial": f"{nan_conf:.0%}",
                "nan_otros"      : f"{nan_otros:.0%}",
                "str_confidencial": f"{str_conf:.0%}" if df[col].dtype == object else "N/A",
                "valores_exclusivos": list(exclusivos)[:5],
            })

    if alertas:
        print("\n⚠️  COLUMNAS QUE DELATAN LA CLASE CONFIDENCIAL:\n")
        for a in sorted(alertas, key=lambda x: x["nivel"], reverse=True):
            print(f"  {a['nivel']}  →  {a['columna']}")
            print(f"           NaN: CONFIDENCIAL={a['nan_confidencial']} | otros={a['nan_otros']}")
            print(f"           Contiene 'CONFIDENCIAL': {a['str_confidencial']}")
            if a["valores_exclusivos"]:
                print(f"           Valores exclusivos: {a['valores_exclusivos']}")
            print()
    else:
        print("\n✅ No se encontraron columnas que delaten la clase CONFIDENCIAL")

    # ── Muestra de registros CONFIDENCIAL ─────────────────────────────────────
    print("\n📋 MUESTRA DE 5 REGISTROS CONFIDENCIAL:")
    print(df_conf.head(5).to_string())

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas para mejorar la predicción."""
    df = df.copy()
    
    f_desparicion = pd.to_datetime(df[['FECHA_DESAPARICION_ANIO', 'FECHA_DESAPARICION_MES', 'FECHA_DESAPARICION_DIA']].assign(DAY=df['FECHA_DESAPARICION_DIA'], MONTH=df['FECHA_DESAPARICION_MES'], YEAR=df['FECHA_DESAPARICION_ANIO'])[['YEAR', 'MONTH', 'DAY']])
    f_registro = pd.to_datetime(df[['FECHA_REGISTRO_ANIO', 'FECHA_REGISTRO_MES', 'FECHA_REGISTRO_DIA']].assign(DAY=df['FECHA_REGISTRO_DIA'], MONTH=df['FECHA_REGISTRO_MES'], YEAR=df['FECHA_REGISTRO_ANIO'])[['YEAR', 'MONTH', 'DAY']])
    
    df["DIAS_LATENCIA_REPORTE"] = (f_registro - f_desparicion).dt.days.fillna(0)
    
    df["ANIOS_ENTRE_EVENTO_Y_REPORTE"] = df["FECHA_REGISTRO_ANIO"] - df["FECHA_DESAPARICION_ANIO"]

    # 2. Edad aproximada (Si existen los datos de nacimiento)
    if "FECHA_NACIMIENTO_ANIO" in df.columns:
        df["EDAD_ESTIMADA"] = (df["FECHA_DESAPARICION_ANIO"] - df["FECHA_NACIMIENTO_ANIO"]).clip(0, 100)
    
    return df

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


def reporte_resultados(y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray = None) -> None:
    """Imprime métricas de evaluación del modelo."""
    acc       = accuracy_score(y_test, y_pred)
    cm        = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    if y_prob is not None:
        # Si es binario, usamos la probabilidad de la clase positiva
        if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        print(f"ROC-AUC  : {auc:.4f}")

def detectar_leakage(X: pd.DataFrame, y: pd.Series, threshold=0.95):
    """Detecta columnas con correlación sospechosamente alta con el target."""
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

def auditoria_columnas(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    y_enc = LabelEncoder().fit_transform(y)
    registros = []

    for col in X.columns:
        try:
            col_enc = LabelEncoder().fit_transform(X[col].fillna("NA").astype(str))
            corr    = abs(np.corrcoef(col_enc, y_enc)[0, 1])
            registros.append({"Variable": col, "Correlación": round(float(corr), 3)})
            print(f"Columna: {col:30s} | Correlación con target: {corr:.4f}")
        except Exception:
            registros.append({"Variable": col, "Correlación": 0.0})

    return pd.DataFrame(registros) 

def main_forecasting():
    data_raw = pd.read_csv(DATA_INPUT)
    target_col = "ESTATUS_VICTIMA"
    data_norm = normaliza_data(data_raw)

    investigar_confidencial(data_norm, target_col=target_col)

    data_norm = data_norm[data_norm[target_col] != "CONFIDENCIAL"].copy()
    
    cols_excluir = [
        target_col, "ID_VICTIMA", "ESTATUS_MAP", "SEXO_MAP", "SEXO",
        "CVE_ENT", "CVE_MUN", "FECHA_NACIMIENTO_CONFIDENCIAL",
        "FECHA_DESAPARICION_CONFIDENCIAL", "FECHA_REGISTRO_CONFIDENCIAL",
    ]

    X = data_norm.drop(columns=cols_excluir, errors="ignore")
    y = data_norm[target_col]

    print("\n--- AUDITORÍA DE COLUMNAS (Pre-procesamiento) ---")
    auditoria_columnas(X, y)
    
    detectar_leakage(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = preprocess_features(X_train)
    X_test = preprocess_features(X_test)
    
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10, 
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    print("\n" + "="*40)
    diagnostico(X_train_scaled, X_test_scaled, y_train, y_test, model)
    
    y_predictions = forecasting(model, X_test_scaled)
    print("\n--- REPORTE DE RESULTADOS FINAL ---")
    y_predictions = model.predict(X_test_scaled)
    
    y_probs = model.predict_proba(X_test_scaled)
    reporte_resultados(y_test, y_predictions, y_probs)

    return y_test, y_predictions, model, X_test, data_raw, X_test_scaled, y_probs


if __name__ == "__main__":
    main_forecasting()