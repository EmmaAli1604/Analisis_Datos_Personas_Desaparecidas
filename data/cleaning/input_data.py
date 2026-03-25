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
    df_processed = pd.read_csv(DATA_PROCESSED)
    
    # Normalizar las fechas
    
    cols_fecha = ['FECHA_NACIMIENTO', 'FECHA_DESAPARICION', 'FECHA_REGISTRO']

    for col in cols_fecha:
        df_processed[col] = df_processed[col].apply(lambda x: limpiar_fecha(x))
    
    # Input data con diferentes técnicas de imputación
    df_mean = input_data_mean(df_processed, ['FECHA_NACIMIENTO', 'FECHA_DESAPARICION', 'FECHA_REGISTRO'])
    df_mean.to_csv('data/input_data/data_mean.csv', index=False)
    df_median = input_data_median(df_processed, ['FECHA_NACIMIENTO', 'FECHA_DESAPARICION', 'FECHA_REGISTRO'])
    df_median.to_csv('data/input_data/data_median.csv', index=False)
    df_mode = input_data_mode(df_processed, ['FECHA_NACIMIENTO', 'FECHA_DESAPARICION', 'FECHA_REGISTRO'])
    
    # Guardar los DataFrames con los datos imputados con estadistica simple
    df_mode.to_csv('data/input_data/data_mode.csv', index=False)
    df_mean.to_csv('data/input_data/data_mean.csv', index=False)
    df_median.to_csv('data/input_data/data_median.csv', index=False)
    
    # Datos imputados para analisis
    df_processed = input_data_mean(df_processed, ['FECHA_NACIMIENTO'])
    # Imputar usando la mediana de la ENTIDAD federativa
    df_processed['FECHA_DESAPARICION'] = df_processed.groupby('ENTIDAD')['FECHA_DESAPARICION'].transform(
        lambda x: x.fillna(x.median())
    )
    df_processed['FECHA_REGISTRO'] = df_processed['FECHA_REGISTRO'].fillna(df_processed['FECHA_DESAPARICION'])
    df_processed.to_csv('data/input_data/data_imputed.csv', index=False)

if __name__ == "__main__":
    main()