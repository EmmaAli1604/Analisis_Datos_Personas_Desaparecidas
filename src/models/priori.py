import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from .normalization import  preprocess_features, normaliza_data
from ..data.config.config import DATA_INPUT

def main():
    print("This is the priori model.")
    data = pd.read_csv(DATA_INPUT)
    
    # 1. Normalizar (fechas, mayúsculas, mapeos) — antes de cualquier acción
    data_norm = normaliza_data(data)
    
    # 2. Nos quedamos con las columnas del dataset que sean útiles para Apriori
    # y modificamos aquellos datos que no sean categóricos y que aportan información.}
    
    #Creamos Rangos para las Edades
    data["EDAD"] = pd.to_datetime(data["FECHA_DESAPARICION"], errors='coerce') - pd.to_datetime(data["FECHA_NACIMIENTO"], errors='coerce')
    data["EDAD"] = data["EDAD"].dt.days // 365

    for i in data["EDAD"].index:
        if data["EDAD"][i] < 12:
            data["EDAD"][i] = "Niño"
        elif data["EDAD"][i] < 18:
            data["EDAD"][i] = "Adolescente"
        elif data["EDAD"][i] < 30:
            data["EDAD"][i] = "Joven"
        elif data["EDAD"][i] < 60:
            data["EDAD"][i] = "Adulto"
        else:
            data["EDAD"][i] = "Adulto Mayor"
            
    # Separamos el año, mes y dia de la desaparición
    data["AÑO_DESAPARICION"] = pd.to_datetime(data["FECHA_DESAPARICION"], errors='coerce').dt.year
    data["MES_DESAPARICION"] = pd.to_datetime(data["FECHA_DESAPARICION"], errors='coerce').dt.month
    data["DIA_SEMANA_DESAPARICION"] = pd.to_datetime(data["FECHA_DESAPARICION"], errors='coerce').dt.dayofweek 
    
    data["TIEMPO_REPORTE"] = pd.to_datetime(data["FECHA_REGISTRO"], errors='coerce') - pd.to_datetime(data["FECHA_DESAPARICION"], errors='coerce')
    data["TIEMPO_REPORTE"] = data["TIEMPO_REPORTE"].dt.days

    # Creamos Rangos para el Tiempo que tardo en colocarse el Reporte
    for i in data["TIEMPO_REPORTE"].index:
        if data["TIEMPO_REPORTE"][i] <= 1:
            data["TIEMPO_REPORTE"][i] = "Inmediato"
        elif data["TIEMPO_REPORTE"][i] <= 7:
            data["TIEMPO_REPORTE"][i] = "Rapido"
        elif data["TIEMPO_REPORTE"][i] <= 30:
            data["TIEMPO_REPORTE"][i] = "Tardio"
        else:
            data["TIEMPO_REPORTE"][i] = "Muy Tardio"

    # Nos quedamos con las columnas que nos interesan para el análisis de Apriori
    # Se considero la Columna Municipio en un inicio pero, además de ser un dato
    # muy específico, tiene una gran cantidad de valores únicos provocando que 
    # el análisis de Apriori consumiera toda la RAM.
    dfFinal = data[['EDAD', 'SEXO', 'AÑO_DESAPARICION', 'MES_DESAPARICION',
                    'TIEMPO_REPORTE', 'ESTATUS_VICTIMA', 'ENTIDAD']]

    # Creamos una lista de transacciones a partir de los datos
    transacciones = []

    for _, row in dfFinal.iterrows():
        trans = [
            f"{row['EDAD']}",
            f"{row['SEXO']}",
            f"{row['AÑO_DESAPARICION']}",
            f"{row['MES_DESAPARICION']}",
            f"{row['TIEMPO_REPORTE']}",
            f"{row['ESTATUS_VICTIMA']}",
            f"{row['ENTIDAD']}"
        ]
        transacciones.append(trans)
    
    te = TransactionEncoder()
    te_ary = te.fit(transacciones).transform(transacciones)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    rasgosFrecuentes = apriori(df_encoded, min_support=0.1, use_colnames=True)
    print("Rasgos Frecuentes Totales:", rasgosFrecuentes.shape[0])
    
    rules = association_rules(
        rasgosFrecuentes,
        metric="confidence",
        min_threshold=0.5
    )

    rules = rules[ (rules['confidence'] >= 0.5) & (rules['lift'] > 1)]

    rules = rules.sort_values(by='lift', ascending=False)

    for i, row in rules.iterrows():
        print(f"{set(row['antecedents'])} → {set(row['consequents'])}")
        print(f"support: {row['support']:.3f}, confidence: {row['confidence']:.3f}, lift: {row['lift']:.3f}")
        print("-----")

if __name__ == "__main__":
    main()