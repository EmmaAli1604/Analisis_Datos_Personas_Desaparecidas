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
        df_copy[col] = (
            df_copy[col]
            .astype(str)
            .str.strip()
            .replace({'CONFIDENCIAL': None, 'nan': None, '': None}) 
        )
        df_copy[col] = pd.to_datetime(
            df_copy[col],
            dayfirst=True,   
            errors='coerce'  
        )

    # Reporte de cuántos nulos quedaron por columna
    for col in cols_fecha:
        nulos = df_copy[col].isna().sum()
        print(f"  {col}: {nulos} valores nulos / no parseados")

    return df_copy

def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el DataFrame para sklearn:
    - Elimina columnas ID/UUID inútiles
    - Encodea categóricas con Label Encoding
    - Convierte fechas a timestamp numérico
    - Elimina columnas que siguen sin poder convertirse
    """
    from sklearn.preprocessing import LabelEncoder

    X = X.copy()

    # Eliminar columnas tipo UUID / ID que no aportan información
    uuid_pattern = r'^[0-9a-fA-F\-]{30,}$'
    for col in X.columns:
        if X[col].dtype == object:
            sample = X[col].dropna().astype(str).head(10)
            if sample.str.match(uuid_pattern).mean() > 0.8:
                print(f"  [DROP] Columna UUID eliminada: {col}")
                X.drop(columns=[col], inplace=True)

    # Convertir fechas a timestamp numérico
    for col in X.columns:
        if X[col].dtype == object:
            try:
                parsed = pd.to_datetime(X[col], errors='raise')
                X[col] = parsed.astype(np.int64) // 10**9  # segundos epoch
                print(f"  [DATE] Columna convertida a timestamp: {col}")
            except Exception:
                pass

    # Encodear columnas categóricas restantes
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].fillna("DESCONOCIDO")
            X[col] = le.fit_transform(X[col].astype(str))
            print(f"  [ENC]  Columna encodeada: {col}")

    # Eliminar cualquier columna que aún no sea numérica (salvaguarda)
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"  [DROP] Columnas no numéricas eliminadas: {non_numeric}")
        X.drop(columns=non_numeric, inplace=True)

    return X