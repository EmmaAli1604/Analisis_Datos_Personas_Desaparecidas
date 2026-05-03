import pandas as pd
import numpy as np

def normaliza_data(df):
    df_copy = df.copy()
    df_copy['ID_VICTIMA'] = df_copy['ID_VICTIMA'].astype(str).str.strip()
    df_copy['ORIGEN_REPORTE'] = df_copy['ORIGEN_REPORTE'].str.upper().str.strip()
    df_copy['SEXO']= df_copy['SEXO'].str.upper().str.strip()
    df_copy['ESTATUS_VICTIMA'] = df_copy['ESTATUS_VICTIMA'].str.upper().str.strip()
    df_copy['ENTIDAD'] = df_copy['ENTIDAD'].str.upper().str.strip()
    df_copy['MUNICIPIO'] = df_copy['MUNICIPIO'].str.upper().str.strip()
    cols_fecha = ['FECHA_NACIMIENTO', 'FECHA_DESAPARICION', 'FECHA_REGISTRO']
    for col in cols_fecha:
        df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')

    return df_copy

def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder

    X = X.copy()
    cols_forzar_str = ["ID_VICTIMA", "ORIGEN_REPORTE"]

    # ── 0. Expandir datetime64 en componentes numéricos ───────────────────────
    cols_fecha = [col for col in X.columns if pd.api.types.is_datetime64_any_dtype(X[col])]
    for col in cols_fecha:
        X[f"{col}_ANIO"] = X[col].dt.year
        X[f"{col}_MES"]  = X[col].dt.month
        X[f"{col}_DIA"]  = X[col].dt.day
        print(f"  [DATE] {col} → {col}_ANIO, {col}_MES, {col}_DIA")

    # Eliminar la columna datetime original (ya expandida)
    X.drop(columns=cols_fecha, inplace=True)

    # ── 1. Eliminar columnas tipo UUID ────────────────────────────────────────
    uuid_pattern = r'^[0-9a-fA-F\-]{30,}$'
    for col in X.columns:
        if col in cols_forzar_str:
            continue
        if X[col].dtype == object:
            sample = X[col].dropna().astype(str).head(10)
            if sample.str.match(uuid_pattern).mean() > 0.8:
                print(f"  [DROP] Columna UUID eliminada: {col}")
                X.drop(columns=[col], inplace=True)

    # ── 2. Convertir strings de fecha a componentes ───────────────────────────
    for col in X.columns:
        if col in cols_forzar_str:
            continue
        if X[col].dtype == object:
            try:
                parsed = pd.to_datetime(X[col], errors='raise')
                X[f"{col}_ANIO"] = parsed.dt.year
                X[f"{col}_MES"]  = parsed.dt.month
                X[f"{col}_DIA"]  = parsed.dt.day
                X.drop(columns=[col], inplace=True)
                print(f"  [DATE] String {col} → {col}_ANIO, {col}_MES, {col}_DIA")
            except Exception:
                pass

    # ── 3. Encodear categóricas ───────────────────────────────────────────────
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == object or col in cols_forzar_str:
            X[col] = X[col].fillna("DESCONOCIDO")
            X[col] = le.fit_transform(X[col].astype(str))
            print(f"  [ENC]  Columna encodeada: {col}")

    # ── 4. Salvaguarda ────────────────────────────────────────────────────────
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"  [DROP] Columnas no numéricas eliminadas: {non_numeric}")
        X.drop(columns=non_numeric, inplace=True)

    return X