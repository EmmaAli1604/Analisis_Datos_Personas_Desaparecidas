import pandas as pd
from datetime import datetime


def corregir_anio_loco(row):
    if pd.isna(row['FECHA_NACIMIENTO']):
        return pd.NaT

    anio = row['FECHA_NACIMIENTO'].year
    mes  = row['FECHA_NACIMIENTO'].month
    dia  = row['FECHA_NACIMIENTO'].day

    anio = _corregir_anio(anio)

    try:
        return datetime(anio, mes, dia)
    except:
        return pd.NaT


def _corregir_anio(anio: int) -> int:

    # Caso A: 2 dígitos (pandas puede parsear "25/01/03" como año 25)
    if anio < 100:
        anio = 2000 + anio if anio <= 26 else 1900 + anio
        return anio

    # Caso B: 3 dígitos (ej: 190, 956, 200)
    if 100 <= anio < 1000:
        str_anio = str(anio)
        if str_anio.startswith('19'):
            # 190 → 1900, 195 → 1950, 199 → 1990  (añadir '0' al final)
            anio = int(str_anio + '0')
        elif str_anio.startswith('20'):
            # 200 → 2000, 202 → 2020
            anio = int(str_anio + '0')
        else:
            # 956 → 1956, 875 → 1875
            anio = 1000 + anio
        return anio

    # Caso C: año futuro lejano con dígitos transpuestos (ej: 3019, 2198)
    if anio > 2100:
        str_anio = str(anio)
        # Intentar reconstruir tomando los últimos 2 dígitos como año dentro del siglo XX/XXI
        sufijo = int(str_anio[-2:])
        anio = 2000 + sufijo if sufijo <= 26 else 1900 + sufijo
        return anio

    # Caso D: año futuro cercano (2027–2100), probablemente centuria equivocada
    if 2026 < anio <= 2100:
        anio = anio - 100
        return anio

    # Año dentro del rango válido (1000–2026): sin cambios
    return anio


def validar_coherencia_fechas(row):
    nac = row['FECHA_NACIMIENTO']
    des = row['FECHA_DESAPARICION']
    reg = row['FECHA_REGISTRO']

    if pd.isna(nac):
        return nac

    ancla = des if pd.notna(des) else reg

    if pd.notna(ancla):
        if nac > ancla:
            if nac.year > ancla.year:
                try:
                    nueva_fecha = nac.replace(year=nac.year - 100)
                    return nueva_fecha if nueva_fecha <= ancla else pd.NaT
                except:
                    return pd.NaT
            else:
                return pd.NaT

    return nac


def limpiar_fecha(x):
    if pd.isna(x) or str(x).strip().upper() == 'CONFIDENCIAL':
        return pd.NaT
    try:
        return pd.to_datetime(x, errors='coerce')
    except:
        return pd.NaT


def normaliza_data(df):
    df_copy = df.copy()

    df_copy['SEXO']= df_copy['SEXO'].str.upper().str.strip()
    df_copy['ESTATUS_VICTIMA'] = df_copy['ESTATUS_VICTIMA'].str.upper().str.strip()
    df_copy['ENTIDAD'] = df_copy['ENTIDAD'].str.upper().str.strip()
    df_copy['MUNICIPIO'] = df_copy['MUNICIPIO'].str.upper().str.strip()

    df_copy['SEXO_MAP'] = df_copy['SEXO'].map({
        'HOMBRE': 1, 'MUJER': 2, 'INDETERMINADO': 3, 'CONFIDENCIAL': 4
    })
    df_copy['ESTATUS_MAP'] = df_copy['ESTATUS_VICTIMA'].map({
        'DESAPARECIDA': 1, 'NO LOCALIZADA': 2, 'CONFIDENCIAL': 3
    })

    cols_fecha = ['FECHA_NACIMIENTO', 'FECHA_DESAPARICION', 'FECHA_REGISTRO']

    for col in cols_fecha:
        df_copy[f'{col}_CONFIDENCIAL'] = df_copy[col].apply(
            lambda x: 1 if str(x).strip().upper() == 'CONFIDENCIAL' else 0
        )

    for col in cols_fecha:
        df_copy[col] = df_copy[col].apply(limpiar_fecha)

    # Primero corregir años malformados, luego validar coherencia entre fechas
    df_copy['FECHA_NACIMIENTO'] = df_copy.apply(corregir_anio_loco, axis=1)
    df_copy['FECHA_NACIMIENTO'] = df_copy.apply(validar_coherencia_fechas, axis=1)

    return df_copy