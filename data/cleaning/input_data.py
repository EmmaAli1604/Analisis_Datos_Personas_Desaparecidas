from ..config.config import DATA_PROCESSED
import pandas as pd

def input_data_mean(df, columns):
    for col in columns:
        df[col] = df[col].fillna(df[col].mean())
    return df

def input_data_median(df, columns):
    for col in columns:
        df[col] = df[col].fillna(df[col].median())
    return df

def input_data_mode(df, columns):
    for col in columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def limpiar_fecha(x):
    if pd.isna(x):
        return pd.NaT
    try:
        return pd.to_datetime(x, errors='coerce')
    except:
        return pd.NaT

def main():
    df_original = pd.read_csv(DATA_PROCESSED)
    
    # Normalizar las fechas una sola vez
    cols_fecha = ['FECHA_NACIMIENTO', 'FECHA_DESAPARICION', 'FECHA_REGISTRO']
    for col in cols_fecha:
        df_original[col] = pd.to_datetime(df_original[col], errors='coerce')
    
    # Input dataset con estadisticas simples para cada columna
    # --- ESCENARIO MEDIA ---
    df_mean = input_data_mean(df_original.copy(), cols_fecha)
    df_mean.to_csv('data/input_data/data_mean.csv', index=False)
    
    # --- ESCENARIO MEDIAN ---
    df_median = input_data_median(df_original.copy(), cols_fecha)
    df_median.to_csv('data/input_data/data_median.csv', index=False)
    
    # --- ESCENARIO MODE ---
    df_mode = input_data_mode(df_original.copy(), cols_fecha)
    df_mode.to_csv('data/input_data/data_mode.csv', index=False)
    
    # Analisis Final
    df_final = df_original.copy()
    
    # Imputación por media en Nacimiento
    df_final['FECHA_NACIMIENTO'] = df_final['FECHA_NACIMIENTO'].fillna(df_final['FECHA_NACIMIENTO'].mean())
    
    # Imputación por mediana de ENTIDAD
    df_final['FECHA_DESAPARICION'] = df_final.groupby('ENTIDAD')['FECHA_DESAPARICION'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Si sigue habiendo nulos (porque toda la entidad es nula), usamos la mediana global
    df_final['FECHA_DESAPARICION'] = df_final['FECHA_DESAPARICION'].fillna(df_final['FECHA_DESAPARICION'].median())
    
    # Relacional: Registro = Desaparición
    df_final['FECHA_REGISTRO'] = df_final['FECHA_REGISTRO'].fillna(df_final['FECHA_DESAPARICION'])
    
    print(df_final.info())
    
    df_final.to_csv('data/input_data/data_imputed.csv', index=False)

if __name__ == "__main__":
    main()