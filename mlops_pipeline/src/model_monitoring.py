
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy import stats
from scipy.spatial import distance

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
# 2. Cargar datos
# ==========================================

@st.cache_data
def load_data():
    # datos de referencia (original)
    df_ref = cargarDatos()
    X_ref = df_ref.drop(columns=['Pago_atiempo'])
    y_ref = df_ref['Pago_atiempo']

    # datos nuevos (simulado con drift de excel)
    df_new = pd.read_excel('../../Base_de_datos_con_Data_Drift_Simulado.xlsx')
    df_new['tendencia_ingresos'] = df_new['tendencia_ingresos'].fillna('Desconocido')
    categorias_validas = ['Creciente', 'Decreciente', 'Estable']
    df_new.loc[~df_new['tendencia_ingresos'].isin(categorias_validas), 'tendencia_ingresos'] = 'Desconocido'
    df_new['fecha_prestamo'] = pd.to_datetime(df_new['fecha_prestamo'])
    df_new['fecha_prestamo'] = df_new['fecha_prestamo'].dt.strftime('%Y%m%d').astype(int)
    X_new = df_new.drop(columns=['Pago_atiempo'])
    y_new = df_new['Pago_atiempo']

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

def psi(col, bins=10):
    ref = X_ref[col].dropna()
    new = X_new[col].dropna()
    breakpoints = np.linspace(min(ref.min(), new.min()), max(ref.max(), new.max()), bins + 1)
    ref_counts = np.histogram(ref, bins=breakpoints)[0] + 1e-6
    new_counts = np.histogram(new, bins=breakpoints)[0] + 1e-6
    ref_pct = ref_counts / ref_counts.sum()
    new_pct = new_counts / new_counts.sum()
    psi_val = np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct))
    return round(psi_val, 4)

def jensen_shannon(col):
    ref = X_ref[col].dropna()
    new = X_new[col].dropna()
    bins = np.linspace(min(ref.min(), new.min()), max(ref.max(), new.max()), 20)
    ref_hist = np.histogram(ref, bins=bins)[0] + 1e-6
    new_hist = np.histogram(new, bins=bins)[0] + 1e-6
    ref_pct = ref_hist / ref_hist.sum()
    new_pct = new_hist / new_hist.sum()
    js = distance.jensenshannon(ref_pct, new_pct)
    return round(js, 4)

def semaforo(pvalue):
    if pvalue < 0.05:
        return "🔴 ALERTA"
    elif pvalue < 0.1:
        return "🟡 PRECAUCION"
    else:
        return "🟢 OK"

def semaforo_psi(valor):
    if valor > 0.2:
        return "🔴 ALERTA"
    elif valor > 0.1:
        return "🟡 PRECAUCION"
    else:
        return "🟢 OK"

def semaforo_js(valor):
    if valor > 0.3:
        return "🔴 ALERTA"
    elif valor > 0.15:
        return "🟡 PRECAUCION"
    else:
        return "🟢 OK"

def tiene_alerta(row):
    estados = [row.get('KS Estado'), row.get('PSI Estado'),
               row.get('JS Estado'), row.get('Chi2 Estado')]
    return "🔴 ALERTA" in estados

def tiene_precaucion(row):
    estados = [row.get('KS Estado'), row.get('PSI Estado'),
               row.get('JS Estado'), row.get('Chi2 Estado')]
    return "🟡 PRECAUCION" in estados and "🔴 ALERTA" not in estados

# ==========================================
# 4. Interfaz Streamlit
# ==========================================

st.title("Monitoreo del Modelo - FitPredict")
st.markdown("Comparacion entre datos de referencia (original) y datos nuevos (simulado con drift) para deteccion de data drift.")

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
        psi_val = psi(col)
        js_val = jensen_shannon(col)
        resultados.append({
            'Variable': col,
            'Tipo': 'Numerica',
            'KS Estadistico': stat,
            'KS P-value': pvalue,
            'KS Estado': semaforo(pvalue),
            'PSI': psi_val,
            'PSI Estado': semaforo_psi(psi_val),
            'Jensen-Shannon': js_val,
            'JS Estado': semaforo_js(js_val),
            'Chi2 Estadistico': None,
            'Chi2 P-value': None,
            'Chi2 Estado': None
        })

    for col in cat_cols:
        try:
            stat, pvalue = chi2_test(col)
            resultados.append({
                'Variable': col,
                'Tipo': 'Categorica',
                'KS Estadistico': None,
                'KS P-value': None,
                'KS Estado': None,
                'PSI': None,
                'PSI Estado': None,
                'Jensen-Shannon': None,
                'JS Estado': None,
                'Chi2 Estadistico': stat,
                'Chi2 P-value': pvalue,
                'Chi2 Estado': semaforo(pvalue)
            })
        except:
            pass

    df_resultados = pd.DataFrame(resultados)
    st.dataframe(df_resultados, use_container_width=True)

    alertas = df_resultados[df_resultados.apply(tiene_alerta, axis=1)]
    precauciones = df_resultados[df_resultados.apply(tiene_precaucion, axis=1)]
    ok = df_resultados[~df_resultados.apply(tiene_alerta, axis=1) &
                       ~df_resultados.apply(tiene_precaucion, axis=1)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Variables en ALERTA", len(alertas))
    col2.metric("Variables en PRECAUCION", len(precauciones))
    col3.metric("Variables OK", len(ok))

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
        psi_val = psi(variable)
        js_val = jensen_shannon(variable)
        st.info(f"KS Test — Estadistico: {stat} | P-value: {pvalue} | Estado: {semaforo(pvalue)}")
        st.info(f"PSI — Valor: {psi_val} | Estado: {semaforo_psi(psi_val)}")
        st.info(f"Jensen-Shannon — Valor: {js_val} | Estado: {semaforo_js(js_val)}")

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
# TAB 3: Recomendaciones str
# ==========================================

with tab3:
    st.subheader("Recomendaciones automaticas")

    if len(alertas) == 0:
        st.success("No se detectaron alertas criticas. El modelo opera dentro de parametros normales.")
    else:
        st.error(f"Se detectaron {len(alertas)} variable(s) con drift significativo:")
        for _, row in alertas.iterrows():
            estado_info = []
            if row.get('KS Estado') == "🔴 ALERTA":
                estado_info.append(f"KS P-value: {row['KS P-value']}")
            if row.get('PSI Estado') == "🔴 ALERTA":
                estado_info.append(f"PSI: {row['PSI']}")
            if row.get('JS Estado') == "🔴 ALERTA":
                estado_info.append(f"Jensen-Shannon: {row['Jensen-Shannon']}")
            if row.get('Chi2 Estado') == "🔴 ALERTA":
                estado_info.append(f"Chi2 P-value: {row['Chi2 P-value']}")
            st.warning(f"Variable **{row['Variable']}** — {', '.join(estado_info)}")

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
            st.info(f"Variable **{row['Variable']}**")