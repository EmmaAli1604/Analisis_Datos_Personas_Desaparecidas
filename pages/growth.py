import streamlit as st
from components.layout import page_layout
import numpy as np
import pandas as pd
import time
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from src.data.config.config import DATA_INPUT

def main():
    page_layout("Explicación a FP-Growth")
    st.title("FP-Growth")
    
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
    
    data["EDAD"] = pd.cut(
        data["EDAD"],
        bins=[-1, 11, 17, 29, 59, 200],
        labels=["Niño", "Adolescente", "Joven", "Adulto", "Adulto Mayor"],
        include_lowest=True
    )
    
    # Fechas
    data["AÑO_DESAPARICION"] = pd.to_datetime(data["FECHA_DESAPARICION"], errors='coerce').dt.year
    data["MES_DESAPARICION"] = pd.to_datetime(data["FECHA_DESAPARICION"], errors='coerce').dt.month
    data["DIA_SEMANA_DESAPARICION"] = pd.to_datetime(data["FECHA_DESAPARICION"], errors='coerce').dt.dayofweek
    
    # Tiempo de Reporte
    data["TIEMPO_REPORTE"] = pd.to_datetime(data["FECHA_REGISTRO"], errors='coerce') - pd.to_datetime(data["FECHA_DESAPARICION"], errors='coerce')
    data["TIEMPO_REPORTE"] = data["TIEMPO_REPORTE"].dt.days
    
    data["TIEMPO_REPORTE"] = pd.cut(
        data["TIEMPO_REPORTE"],
        bins=[-np.inf, 1, 7, 30, np.inf],
        labels=["Inmediato", "Rápido", "Tardío", "Muy Tardío"],
        include_lowest=True
    )
    
    # =========================
    #   PREPARAR TRANSACCIONES
    # =========================
    
    dfFinal = data[['EDAD', 'SEXO', 'AÑO_DESAPARICION', 'MES_DESAPARICION', 'TIEMPO_REPORTE', 
                    'ESTATUS_VICTIMA', 'ENTIDAD']]
    
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
    #   EJECUTAR FP-GROWTH
    # =========================
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("FP-Growth")
        inicio_fpg = time.time()
        rasgosFrecuentes_fpgrowth = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
        fin_fpg = time.time()
        tiempo_fpg = fin_fpg - inicio_fpg
        
        st.metric("Itemsets Frecuentes", rasgosFrecuentes_fpgrowth.shape[0])
        st.metric("Tiempo de ejecución", f"{tiempo_fpg:.4f}s")
    
    with col2:
        st.subheader("Apriori (Comparación)")
        inicio_apr = time.time()
        rasgosFrecuentes_apriori = apriori(df_encoded, min_support=min_support, use_colnames=True)
        fin_apr = time.time()
        tiempo_apr = fin_apr - inicio_apr
        
        st.metric("Itemsets Frecuentes", rasgosFrecuentes_apriori.shape[0])
        st.metric("Tiempo de ejecución", f"{tiempo_apr:.4f}s")
    
    # =========================
    #   ITEMSETS FRECUENTES
    # =========================
    
    st.markdown("---")
    st.subheader("Top Itemsets Frecuentes")
    
    rasgosFrecuentes_sorted = rasgosFrecuentes_fpgrowth.sort_values('support', ascending=False)
    
    st.write("**Top 15 Itemsets**")
    display_df = rasgosFrecuentes_sorted.head(15).copy()
    display_df['itemsets'] = display_df['itemsets'].astype(str)
    display_df['support'] = display_df['support'].round(4)
    st.dataframe(display_df, use_container_width=True)
    
    # =========================
    #   REGLAS DE ASOCIACIÓN
    # =========================
    
    st.markdown("---")
    st.subheader("Reglas de Asociación")
    
    rules = association_rules(rasgosFrecuentes_fpgrowth, metric="confidence", min_threshold=min_conf)
    rules = rules[(rules['confidence'] >= min_conf) & (rules['lift'] > min_lift)]
    rules = rules.sort_values(by='lift', ascending=False)
    
    if len(rules) > 0:
        st.metric("Total de Reglas", len(rules))
        
        st.write("**Top 20 Reglas por Lift**")
        display_rules = rules.head(20).copy()
        display_rules['antecedents'] = display_rules['antecedents'].astype(str)
        display_rules['consequents'] = display_rules['consequents'].astype(str)
        display_rules = display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        display_rules.columns = ['Antecedentes', 'Consecuentes', 'Support', 'Confidence', 'Lift']
        
        for col in ['Support', 'Confidence', 'Lift']:
            display_rules[col] = display_rules[col].round(4)
        
        st.dataframe(display_rules, use_container_width=True)
    
    else:
        st.warning("No se encontraron reglas con los parámetros seleccionados")
    
    # =========================
    #   ESTADÍSTICAS
    # =========================
    
    st.markdown("---")
    st.subheader("Estadísticas del Dataset")
    st.metric("Total de Registros", f"{len(data):,}")

if __name__ == "__main__":
    main()