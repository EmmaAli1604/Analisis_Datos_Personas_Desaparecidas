from ..config.config import DATA_RAW
import pandas as pd

def load_data():
    df = pd.read_csv(DATA_RAW)
    return df

def print_data_info(df):
    print("="*80)
    print("Data loaded successfully.")
    print("="*80)
    print(df.head())
    print("="*80)
    print("Data Info:")
    print(df.info())
    print("="*80)
    print("Data Description:")
    print(df.describe())
    print("="*80)
    print("Missing Values:")
    print(df.isnull().sum())
    print("="*80)

def limpiar_fecha(x):
    if pd.isna(x) or str(x).strip().upper() == 'CONFIDENCIAL':
        return pd.NaT
    try:
        # Intentamos la conversión estándar
        return pd.to_datetime(x, errors='coerce')
    except:
        return pd.NaT

cols_fecha = ['FECHA_NACIMIENTO', 'FECHA_DESAPARICION', 'FECHA_REGISTRO']
    
def main():
    df_raw = load_data()
    print_data_info(df_raw)
    df_raw_copy = df_raw.copy()

    # Columna con un mapeo de hombre, mujer, indeterminado a 1, 2, 3 respectivamente
    df_raw_copy['SEXO_MAP'] = df_raw_copy['SEXO'].map({'HOMBRE': 1, 'MUJER': 2, 'INDETERMINADO': 3, 'CONFIDENCIAL': 4})
    
    # Columna con un mapeo de estatus victima desparecida, no localizada, confidencial a 1, 2, 3 respectivamente
    df_raw_copy['ESTATUS_MAP'] = df_raw_copy['ESTATUS_VICTIMA'].map({'DESAPARECIDA': 1, 'NO LOCALIZADA': 2, 'CONFIDENCIAL': 3})
    
    cols_fecha = ['FECHA_NACIMIENTO', 'FECHA_DESAPARICION', 'FECHA_REGISTRO']
    
    # Se agrega una columna de control para cada fecha indicando si es confidencial o no
    for col in cols_fecha:
        df_raw_copy[f'{col}_CONFIDENCIAL'] = df_raw_copy[col].apply(
            lambda x: 1 if str(x).strip().upper() == 'CONFIDENCIAL' else 0
        )
    
    # Normalizar las fechas convirtiendo CONFIDENCIAL a NaT y los demás valores a formato datetime
    for col in cols_fecha:
        df_raw_copy[col] = df_raw_copy[col].apply(lambda x: limpiar_fecha(x))
    
    df_processed = df_raw_copy.copy()
    print_data_info(df_processed)
    
    df_processed.to_csv("data/processed/data_processed.csv", index=False)
    
if __name__ == "__main__":
    main()