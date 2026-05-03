import streamlit as st
from components.layout import page_layout
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from src.data.config.config import DATA_INPUT

def main():
    page_layout("Explicacion a Priori")
    st.title("Apriori")
    # Carga del Archivo
    data = pd.read_csv(DATA_INPUT)
    
    # Parametros
    st.sidebar.header("Parámetros")
    min_support = st.sidebar.slider("Min Support", 0.01, 0.5, 0.1)
    min_conf = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.5)
    min_lift = st.sidebar.slider("Min Lift", 1.0, 5.0, 1.0)
    
    # =========================
    #   PREPROCESAMIENTO
    # =========================
    
    # Edades
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
            
    # Fechas
    data["AÑO_DESAPARICION"] = pd.to_datetime(data["FECHA_DESAPARICION"], errors='coerce').dt.year
    data["MES_DESAPARICION"] = pd.to_datetime(data["FECHA_DESAPARICION"], errors='coerce').dt.month
    data["DIA_SEMANA_DESAPARICION"] = pd.to_datetime(data["FECHA_DESAPARICION"], errors='coerce').dt.dayofweek 
    
    # Tiempo de Reporte
    data["TIEMPO_REPORTE"] = pd.to_datetime(data["FECHA_REGISTRO"], errors='coerce') - pd.to_datetime(data["FECHA_DESAPARICION"], errors='coerce')
    data["TIEMPO_REPORTE"] = data["TIEMPO_REPORTE"].dt.days

    for i in data["TIEMPO_REPORTE"].index:
        if data["TIEMPO_REPORTE"][i] <= 1:
            data["TIEMPO_REPORTE"][i] = "Inmediato"
        elif data["TIEMPO_REPORTE"][i] <= 7:
            data["TIEMPO_REPORTE"][i] = "Rapido"
        elif data["TIEMPO_REPORTE"][i] <= 30:
            data["TIEMPO_REPORTE"][i] = "Tardio"
        else:
            data["TIEMPO_REPORTE"][i] = "Muy Tardio"

    # Selección columnas
    dfFinal = data[['EDAD', 'SEXO', 'AÑO_DESAPARICION', 'MES_DESAPARICION',
                    'TIEMPO_REPORTE', 'ESTATUS_VICTIMA', 'ENTIDAD']]
    
    st.subheader("Datos procesados")
    st.dataframe(dfFinal.head())

    # =========================
    # TRANSACCIONES
    # =========================
    transacciones = []

    for _, row in dfFinal.iterrows():
        trans = [
            f"EDAD_{row['EDAD']}",
            f"SEXO_{row['SEXO']}",
            f"AÑO_{row['AÑO_DESAPARICION']}",
            f"MES_{row['MES_DESAPARICION']}",
            f"TIEMPO_{row['TIEMPO_REPORTE']}",
            f"ESTATUS_{row['ESTATUS_VICTIMA']}",
            f"ENTIDAD_{row['ENTIDAD']}"
        ]
        transacciones.append(trans)
    
    te = TransactionEncoder()
    te_ary = te.fit(transacciones).transform(transacciones)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # =========================
    # APRIORI
    # =========================
    if st.button("Generar reglas"):
        rasgosFrecuentes = apriori(df_encoded, min_support=min_support, use_colnames=True)
        print("Rasgos Frecuentes Totales:", rasgosFrecuentes.shape[0])
    
        rules = association_rules(
            rasgosFrecuentes,
            metric="confidence",
            min_threshold=min_conf
        )
        
        # Filtrado
        rules = rules[ (rules['confidence'] >= min_conf) & (rules['lift'] > min_lift)]

        if rules.empty:
                st.warning("No se encontraron reglas con esos parámetros.")
        else:
            st.success(f"{len(rules)} reglas encontradas")
            
            rules = rules.sort_values(by='lift', ascending=False)
            
            rules["Regla"] = rules.apply(lambda row: f"{set(row['antecedents'])} → {set(row['consequents'])}", axis=1)
            st.dataframe(
                    rules[["Regla", "support", "confidence", "lift"]],
                    use_container_width=True
                )
            
            items = list(df_encoded.columns)
            
            st.sidebar.subheader("Filtro por atributo")

            atributo = st.sidebar.selectbox("Selecciona atributo", ["Todos"] + items)

            tipo_filtro = st.sidebar.selectbox(
                "Dónde buscar",
                ["Ambos", "Antecedente", "Consecuente"]
            )
            
            if atributo != "Todos":
                if tipo_filtro == "Antecedente":
                    rules = rules[rules['antecedents'].apply(lambda x: atributo in x)]
                elif tipo_filtro == "Consecuente":
                    rules = rules[rules['consequents'].apply(lambda x: atributo in x)]
                else:  # Ambos
                    rules = rules[
                        rules['antecedents'].apply(lambda x: atributo in x) |
                        rules['consequents'].apply(lambda x: atributo in x)
                    ]

main()