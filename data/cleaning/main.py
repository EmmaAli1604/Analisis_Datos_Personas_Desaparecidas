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
    
def main():
    df_raw = load_data()
    df_clean = df_raw.copy()
    # Normalizar las fechas
    df_clean['FECHA_NACIMIENTO'] = pd.to_datetime(df_clean['FECHA_NACIMIENTO'], errors='coerce')
    df_clean['FECHA_DESAPARICION'] = pd.to_datetime(df_clean['FECHA_DESAPARICION'], errors='coerce')
    df_clean['FECHA_REGISTRO'] = pd.to_datetime(df_clean['FECHA_REGISTRO'], errors='coerce')
    
    # Columna con un mapeo de hombre, mujer, indeterminado a 1, 2, 3 respectivamente
    df_clean['SEXO_MAP'] = df_clean['SEXO'].map({'HOMBRE': 1, 'MUJER': 2, 'INDETERMINADO': 3})
    
    # Columna con un mapeo de estatus victima desparecida, no localizada, confidencial a 1, 2, 3 respectivamente
    df_clean['ESTATUS_MAP'] = df_clean['ESTATUS_VICTIMA'].map({'DESAPARECIDA': 1, 'NO LOCALIZADA': 2, 'CONFIDENCIAL': 3})
    
    # Se elimina los registros sin fecha de registro
    df_clean = df_clean.dropna(subset=['FECHA_REGISTRO'])

if __name__ == "__main__":
    main()