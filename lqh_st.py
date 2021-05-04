# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 18:48:58 2020

@author: mcuevas
"""

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from empiricaldist import Pmf
from empiricaldist import Cdf

from PIL import Image


################################### CARGA DE DATA #######################################################################
#@st.cache()
def load_data():
    data=pd.read_csv('LQH_ST.csv')
    #data=data[data.Zona!='Otro']
    return data
lqh=load_data()

fec_ini=lqh.Fecha.min()
fec_fin=lqh.Fecha.max()

prime=lqh[lqh.Franja=="Prime"]
offprime=lqh[lqh.Franja=="Off Prime PM"]
prime2=lqh[lqh.Franja=="Prime 2da"]
off2=lqh[lqh.Franja=="Off Prime AM"]


hogar_canales=['SH_C13','SH_TVN','SH_MEGA','SH_CHV','SH_TVPAGO']
funciones=['mean','count','median','var','std']


###################### IMAGEN y TEXTO INICIAL ##########################################################################
img=Image.open("lqh.png")


st.title("Lugares que Hablan Análisis de Datos")
st.image(img)
st.markdown("Esta aplicación fue realizada por el equipo *BI-Comercial de Canal 13* tomando como fuente de datos la información proporcionada por el área de programación, cualquier duda escribir al correo **<equipobi@13.cl>**")


st.markdown("Los datos contemplados van desde el **%s** al **%s** con un total de **%i** emisiones"%(fec_ini,fec_fin,len(lqh)))
st.write("""
        No están contemplados los capítulos donde Pancho Saavedra no estuvo en el programa, tampoco los resumenes de capítulos, ni las emisiones de los capítulos Extranjeros.
        """
        )
################################# Mostrar Tabla y Caracteríticas     #######################################################

st.sidebar.title("Navegación")

if st.sidebar.checkbox("Base Lugares que Hablan",True):
    st.dataframe(lqh)

########################### Gráficos PRIME y OFF        ######################################################################

salida_Franja=lqh.groupby(['Franja'])['SH_C13','SC_C13'].agg(funciones)
salida_Franja=salida_Franja.reset_index()
#valor_x=list(salida_Franja.index)
st.write(salida_Franja)
#st.write(salida_Franja.columns)

select=st.sidebar.selectbox('Evolucion Franjas',['Prime','Off Prime PM','Prime Segunda Franja','Off Prime AM'],key='1')
if st.sidebar.checkbox("Mostrar",True):
    
    st.markdown("## Rendimiento Share Hogar %s"%(select))
                
    if select=='Prime':
        fig=px.line(prime,x='Fecha',y=hogar_canales)
    elif select=='Prime Segunda Franja':
        fig=px.line(prime2,x='Fecha',y=hogar_canales)
    elif select=='Off Prime PM':
        fig=px.line(offprime,x='Fecha',y=hogar_canales)
    else:
        fig=px.line(off2,x='Fecha',y=hogar_canales)
    
    st.plotly_chart(fig)
    
   # fig_bar=px.bar(salida_Franja,x='Franja',y=('SH_C13','mean'))
   # st.plotly_chart(fig_bar)
##########################   FUNCIONES DE DISTRIBUCION   ######################################################################
cdf_p=Cdf.from_seq(prime['SH_C13'])
cdf_o=Cdf.from_seq(offprime['SH_C13'])

x=np.array(cdf_p.index)
y=cdf_p.values

min_x=int(np.around(x.min()))
max_x=int(np.around(x.max()))

share_min=st.sidebar.slider("Share Hogar Minimo ",min_x,max_x)
share_max=st.sidebar.slider("Share Hogar Maximo ",min_x,max_x)

#probabilidad_1=round((cdf(share_cdf_2))*100,1)
#probabilidad_2=round((1-cdf(share_cdf_2))*100,1)

st.markdown("## Probabilidad Share Hogar*")
st.write("(*)Para este análisis solo se consideran datos históricos y el cálculo es con la curva real")
           
if share_max>share_min:
    probabilidad_p=round((cdf_p(share_max)-cdf_p(share_min))*100,1)
    probabilidad_o=round((cdf_o(share_max)-cdf_o(share_min))*100,1)
    st.markdown("La probabilidad de tener un **Share Hogar en el Prime** entre %i y %i es de %f porciento"%(share_min,share_max,probabilidad_p))
    st.markdown("La probabilidad de tener un **Share Hogar en el OFF Prime** entre %i y %i es de %f porciento"%(share_min,share_max,probabilidad_o))
#elif share_min==min_x:
#    probabilidad=round((1-cdf(share_max))*100,1)
#    st.markdown("La probabilidad de tener un Share Hogar en el Prime >= %i es de %f porciento"%(share_max,probabilidad))
else:
    st.markdown("El valor del share maximo debe ser mayor que el share minimo")

np.random.seed(42)
mu_p=np.mean(prime.SH_C13)
sigma_p=np.std(prime.SH_C13)

mu_o=np.mean(offprime.SH_C13)
sigma_o=np.std(offprime.SH_C13)

teorico_p=np.random.normal(mu_p,sigma_p,size=10000)
teorico_o=np.random.normal(mu_o,sigma_o,size=10000)
              
             
fig=plt.figure(figsize=(12,6))
sns.kdeplot(prime.SH_C13,color='b',label='Real Prime',shade=True)
sns.kdeplot(teorico_p,color='r',label="Teorico Prime")

sns.kdeplot(offprime.SH_C13,color='y',label='Real OFF',shade=True)
sns.kdeplot(teorico_o,color='c',label="Teorico OFF")

#plt.fill_between(x[x>=share_cdf],y[x>=share_cdf],color='c')
plt.ylabel('PDF')
plt.xlabel('Share Hogar LQH')
st.pyplot(fig)
