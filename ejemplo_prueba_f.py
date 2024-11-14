import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from prophet import Prophet
import datetime
from datetime import datetime, date, timedelta
import pickle

# cargar logo
logo = plt.imread(r"C:\Users\AxelC\Downloads\LOGOS_NL_SIMA_FINAL.png") ############################################## Esto se cambia
st.sidebar.image(logo)

# nombres estaciones 
nombres = ['SURESTE','NORESTE','CENTRO','NOROESTE','SUROESTE','NOROESTE2',
 'NORTE','SUROESTE2','SURESTE2','SURESTE3','SUR','NORTE2','NORESTE2',
 'NORESTE3','NOROESTE3']

with st.sidebar:
    estacion = st.selectbox('Escoga una estación',nombres)  
    begin_time = st.date_input("Fecha inicio", datetime(2023,8,17))
    end_time = st.date_input("Fecha final", datetime(2023,8,20)) 


def filter_data(estacion):
    if estacion:
        df_copy = pd.read_excel(f"C:\\Users\\AxelC\\Downloads\\DATOS_LIMPIOS\\{estacion}.xlsx") ############################################## Esto se cambia
    
        
    return df_copy

df_ = filter_data(estacion)
df_ = df_.drop(columns=['CO', 'NO', 'NO2', 'NOX', 'PM10', 'PM2.5', 'PRS', 'RAINF',
       'RH', 'SO2', 'SR', 'TOUT', 'WSR', 'WDR'])
df_['weeknum'] = df_['date'].dt.isocalendar().week

st.title(f"Predicción de O3: Estación {estacion}")

# Grafica de prediccion
st.subheader("Visualización Predicción")


def prophet_f(df_, end_time, estacion):
    if estacion and end_time:

        pkl_path = f"C:\\Users\\AxelC\\Downloads\\PROPHETS_LIMPIOS\\{estacion}.pkl" ############################################## Esto se cambia

        m = pd.read_pickle(pkl_path)

        # resta de dias del ultimo dato hasta el final
        dias = (datetime(end_time.year,end_time.month, end_time.day) - datetime(2023, 8, 17)).days

        future = m.make_future_dataframe(periods=dias*24, freq="1H")
        forecast = m.predict(future)  
        return forecast

print(df_)
forecast = prophet_f(df_, end_time, estacion)
forecast_fut = forecast.copy()
forecast_fut = forecast_fut[(forecast_fut["ds"] >= pd.to_datetime(begin_time)) & (forecast_fut["ds"] <= pd.to_datetime(end_time))]

st.bar_chart(
   forecast_fut, x="ds", y="yhat", color="#b4864c"
)


forecast_fut['date'] = pd.to_datetime(forecast_fut['ds'])

st.subheader("Comparación Datos Históricos vs. Predicción")
st.markdown("Datos correpsondientes al mismo número de semana y día semana con el día a predecir")

forecast_fut['dia_semana'] = forecast_fut['date'].dt.dayofweek
forecast_fut['weeknum'] = forecast_fut['date'].dt.isocalendar().week
forecast_fut['hora'] = forecast_fut['date'].dt.hour
forecast_fut['O3 predicción'] = forecast_fut['yhat']


newdf = df_.drop_duplicates(subset = ['weeknum', 'dia_semana', 'hora'], keep = 'last').reset_index(drop = True)
newdf = newdf.sort_values(by=['weeknum', 'dia_semana', 'hora'], ascending=True)
newdf = pd.merge(forecast_fut, newdf, on=['weeknum', 'dia_semana', 'hora'], how='left')
st.line_chart(
   newdf, x="ds", y=["O3", 'O3 predicción'], color=[(76,184,117), "#b4864c"]
)

# Datos historicos
st.subheader("Visualización Históricos")
st.line_chart(
   df_, x="date", y="O3", color=(76,184,117)
)

forecast_epam = forecast[forecast['ds'] < '2023-08-18']
df_junto = pd.DataFrame()
df_junto['date'] = df_.loc[:, 'date']
df_junto['O3'] = df_.loc[:, 'O3']
df_junto['O3 predicción'] = forecast_epam.loc[:, 'yhat']

# MOdelo vs Historicos
st.subheader("Visualización Históricos y Modelo")
st.line_chart(
   df_junto, x="date", y=["O3", "O3 predicción"], color=[(76,184,117), "#b4864c"]  
)



# Datos historicos
st.subheader("Comportamiento y Tendencia")
promedio_o3_db = df_['O3'].mean()
promedio_o3_db_ph = forecast['yhat'].mean()


epam = (df_['O3'].values - forecast_epam['yhat'].values) / forecast_epam['yhat'].values
epam = abs(epam).mean()

col1, col2, col3 = st.columns(3)
col1.metric("Promedio histórico de O3 (ppb)", f"{promedio_o3_db:,.3f}")
col2.metric("Promedio futuro de O3 (ppb)", f"{promedio_o3_db_ph:,.3f}")
col3.metric("EPAM (%)", f"{epam:,.2f}")

# Mostrar tabla de datos
st.subheader("Predicción")
st.dataframe(forecast)

# Descargar prediccion
st.subheader("Descargar predicción")
st.cache_data()

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(forecast)

date_str = end_time.strftime('%Y-%m-%d')
st.download_button(
    label="Descargar predicción",
    data=csv,
    file_name=f'{estacion}_prediccion_{date_str}.csv', ################ Ponerle estacion - fecha - prediccion
    mime='text/csv',
)
st.cache_data.clear()