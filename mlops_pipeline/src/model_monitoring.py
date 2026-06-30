import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy import stats
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Monitoreo del Modelo", layout="wide")

# ==========================================
# 1. Funcion para cargar datos
# ==========================================

def cargarDatos():
    df = pd.read_csv('../../Base_de_datos.csv', sep=';')

    df['tendencia_ingresos'] = df['tendencia_ingresos'].fillna('Desconocido')
    categorias_validas = ['Creciente', 'Decreciente', 'Estable']
    df.loc[~df['tendencia_ingresos'].isin(categorias_validas), 'tendencia_ingresos'] = 'Desconocido'

    df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'], format='%d/%m/%Y %H:%M')
    df['fecha_prestamo'] = df['fecha_prestamo'].dt.strftime('%Y%m%d').astype(int)

    return df

# ==========================================
# 2. Cargar y dividir datos
# ==========================================

@st.cache_data
def load_data():
    df = cargarDatos()
    X = df.drop(columns=['Pago_atiempo'])
    y = df['Pago_atiempo']
    X_ref, X_new, y_ref, y_new = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_ref, X_new, y_ref, y_new

X_ref, X_new, y_ref, y_new = load_data()

# ==========================================
# 3. Funciones de calculo de drift
# ==========================================

def ks_test(col):
    stat, pvalue = stats.ks_2samp(X_ref[col].dropna(), X_new[col].dropna())
    return round(stat, 4), round(pvalue, 4)

def chi2_test(col):
    categorias = set(X_ref[col].dropna().unique()) | set(X_new[col].dropna().unique())
    ref_counts = X_ref[col].value_counts().reindex(categorias, fill_value=0)
    new_counts = X_new[col].value_counts().reindex(categorias, fill_value=0)
    stat, pvalue = stats.chi2_contingency(
        pd.DataFrame([ref_counts, new_counts])
    )[:2]
    return round(stat, 4), round(pvalue, 4)

def semaforo(pvalue):
    if pvalue < 0.05:
        return "🔴 ALERTA"
    elif pvalue < 0.1:
        return "🟡 PRECAUCION"
    else:
        return "🟢 OK"

# ==========================================
# 4. Interfaz Streamlit
# ==========================================

st.title("Monitoreo del Modelo - FitPredict")
st.markdown("Comparacion entre datos de referencia (train) y datos nuevos (test) para deteccion de data drift.")

# --- Tabs principales ---
tab1, tab2, tab3 = st.tabs(["Resumen de Drift", "Analisis por Variable", "Recomendaciones"])

# ==========================================
# TAB 1: Resumen de drift
# ==========================================

with tab1:
    st.subheader("Tabla resumen de data drift")

    num_cols = X_ref.select_dtypes('number').columns.tolist()
    cat_cols = X_ref.select_dtypes('object').columns.tolist()

    resultados = []

    for col in num_cols:
        stat, pvalue = ks_test(col)
        resultados.append({
            'Variable': col,
            'Tipo': 'Numerica',
            'Test': 'KS',
            'Estadistico': stat,
            'P-value': pvalue,
            'Estado': semaforo(pvalue)
        })

    for col in cat_cols:
        try:
            stat, pvalue = chi2_test(col)
            resultados.append({
                'Variable': col,
                'Tipo': 'Categorica',
                'Test': 'Chi2',
                'Estadistico': stat,
                'P-value': pvalue,
                'Estado': semaforo(pvalue)
            })
        except:
            pass

    df_resultados = pd.DataFrame(resultados)
    st.dataframe(df_resultados, use_container_width=True)

    alertas = df_resultados[df_resultados['Estado'] == "🔴 ALERTA"]
    precauciones = df_resultados[df_resultados['Estado'] == "🟡 PRECAUCION"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Variables en ALERTA", len(alertas))
    col2.metric("Variables en PRECAUCION", len(precauciones))
    col3.metric("Variables OK", len(df_resultados) - len(alertas) - len(precauciones))

# ==========================================
# TAB 2: Analisis por variable
# ==========================================

with tab2:
    st.subheader("Comparacion de distribucion por variable")

    variable = st.selectbox("Selecciona una variable:", num_cols + cat_cols)

    if variable in num_cols:
        df_plot = pd.DataFrame({
            'valor': pd.concat([X_ref[variable], X_new[variable]]),
            'dataset': ['Referencia'] * len(X_ref) + ['Nuevo'] * len(X_new)
        })
        fig = px.histogram(df_plot, x='valor', color='dataset', barmode='overlay',
                          title=f'Distribucion de {variable}',
                          opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

        stat, pvalue = ks_test(variable)
        st.info(f"KS Test — Estadistico: {stat} | P-value: {pvalue} | Estado: {semaforo(pvalue)}")

    else:
        ref_pct = X_ref[variable].value_counts(normalize=True).reset_index()
        ref_pct.columns = [variable, 'proporcion']
        ref_pct['dataset'] = 'Referencia'

        new_pct = X_new[variable].value_counts(normalize=True).reset_index()
        new_pct.columns = [variable, 'proporcion']
        new_pct['dataset'] = 'Nuevo'

        df_cat = pd.concat([ref_pct, new_pct])
        fig = px.bar(df_cat, x=variable, y='proporcion', color='dataset',
                    barmode='group', title=f'Distribucion de {variable}')
        st.plotly_chart(fig, use_container_width=True)

        try:
            stat, pvalue = chi2_test(variable)
            st.info(f"Chi2 Test — Estadistico: {stat} | P-value: {pvalue} | Estado: {semaforo(pvalue)}")
        except:
            st.warning("No se pudo calcular el test para esta variable.")

# ==========================================
# TAB 3: Recomendaciones
# ==========================================

with tab3:
    st.subheader("Recomendaciones automaticas")

    if len(alertas) == 0:
        st.success("No se detectaron alertas criticas. El modelo opera dentro de parametros normales.")
    else:
        st.error(f"Se detectaron {len(alertas)} variable(s) con drift significativo:")
        for _, row in alertas.iterrows():
            st.warning(f"Variable **{row['Variable']}** — P-value: {row['P-value']}")

        st.markdown("""
        ### Acciones sugeridas:
        - Revisar la fuente de datos para las variables en alerta
        - Evaluar si es necesario **reentrenar el modelo** con datos mas recientes
        - Analizar si hubo cambios en el proceso de captura de datos
        - Considerar agregar estas variables al proceso de validacion periodica
        """)

    if len(precauciones) > 0:
        st.warning(f"{len(precauciones)} variable(s) en zona de precaucion — monitorear de cerca:")
        for _, row in precauciones.iterrows():
            st.info(f"Variable **{row['Variable']}** — P-value: {row['P-value']}")