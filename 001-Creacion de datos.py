
######################################################################
#      PROYECTO: MODELO PARA IDENTIFICAR LOS CLIENTES MÁS PROPENSO
#                   A LA FUGA CON MACHINE LEARNING
#####################################################################

#######################################
# Autor: Nahuel Canelo
# Correo: nahuelcaneloaraya@gmail.com
#######################################

########################################
# IMPORTAMOS LAS LIBRERIAS DE INTERÉS
########################################

import numpy as np
import pandas as pd
import uuid
import random
from scipy.stats import beta
from numpy.random import rand
import warnings
warnings.filterwarnings('once')


#############################################
# CREAMOS LAS FUNCIONES QUE VAMOS A UTILIZAR
#############################################

# Construiremos una función que defina el comportamiento historico del cliente según si su marca de fuga es 0 o 1

# Rezagos: Número de meses anteriores al mes de referencia con información, 1 es el mes más cercano y 12 es el más lejano
# Valor_max: Es el valor máx alcanzado por la variable historica que se desea construir

# Si el cliente se va a fugar, se construyen variables hisoticas a partir de una función beta decreciente
# Si el cliente no se va a fugar, se construyen variables historicas a partir de una función normal

def tendencia(x, rezagos,valor_max):
    ds=valor_max/100# ds
    noise = np.random.uniform(-1, 1, rezagos)  # Se utilizará para agregar ruido
    if x==1: # Distribución beta
        a, b, inicio, fin = 10, 2, 0.1, 0.99  # párametros de la distribución beta
        x = np.linspace(beta.ppf(inicio, a, b), beta.ppf(fin, a, b), 12)
        x_valores = beta.pdf(x, a, b)
        x_valores= (x_valores/np.max(x_valores)) * valor_max * rand(1)
        x_valores = x_valores + (x_valores * noise)
    else: # Distribución normal
        x_valores = np.random.normal(valor_max*rand(1), ds * rand(1), rezagos)
        x_valores = x_valores + (x_valores * noise)
        x_valores = (x_valores/np.max(x_valores)) * valor_max * rand(1)
    return pd.Series(x_valores)

###########################################################################
# CREAMOS LA BASE DE DATOS SEGÚN TENDENCIAS PLAUSIBLES ASOCIADAS A LA FUGA
###########################################################################

n=1000 # Número de clientes/registros que se van a crear
rezagos=12 # Número de meses anteriores al mes de referencia para los cuales se crearán variables

#Inicializamos un dataframe con los ID de cada cliente
data=pd.DataFrame({"ID": [uuid.uuid4() for _ in range(n)]})

# Agregamos una tasa de fuga del 15% (FUGA: 1: NO compra el siguiente mes, 0: SI lo hace)
data=data.assign(fuga=random.choices([1,0], cum_weights=(15,85),k = n))

# Monto mensual de compras de cada cliente
data_monto=data["fuga"].apply(tendencia,args=(rezagos, 50000))
data_monto.columns=np.array(["monto_"+str(i+1) for i in range(rezagos)])
data=pd.concat([data,data_monto],axis=1)

# Indicador mensual de satisfacción de cada cliente
data_satisfaccion=data["fuga"].apply(tendencia,args=(rezagos, 7))
data_satisfaccion.columns=np.array(["satisfaccion_"+str(i+1) for i in range(rezagos)])
data=pd.concat([data,data_satisfaccion],axis=1)

# Número de productos mensuales distintos en la canasta de cada cliente
data_canasta = data["fuga"].apply(tendencia, args=(rezagos, 15))
data_canasta.columns = np.array(["canasta_" + str(i + 1) for i in range(rezagos)])
data = pd.concat([data, data_canasta], axis=1)

# Número de dias entre la solicitud del pedido y la entrega para cada cliente
data_espera = data["fuga"].apply(tendencia, args=(rezagos, 20))
data_espera.columns = np.array(["espera_" + str(i+1) for i in reversed(range(rezagos))])
data = pd.concat([data, data_espera], axis=1)

# Antiguedad del cliente (meses)
data=data.assign(antiguedad= np.random.uniform(15,300))

# Tipo de cliente
data=data.assign(grande=random.choices([1,0], cum_weights=(30,70),k = n))
data=data.assign(pequeno= lambda x: (x.grande==0) * 1)


##############################
# VISUALIZAMOS ALGUNOS CASOS
##############################

data_monto=data.query("fuga == 1")[['ID','monto_1', 'monto_2', 'monto_3', 'monto_4', 'monto_5',
       'monto_6', 'monto_7', 'monto_8', 'monto_9', 'monto_10', 'monto_11',
       'monto_12']][0:3].copy()
data_monto["ID"]=range(data_monto.shape[0])
data_monto_melt=pd.melt(data_monto,id_vars='ID')


sns.set(style='darkgrid',context='paper',font_scale=1.2,palette='colorblind')

a4_dims = (50, 8.27)
fig1=sns.lineplot(data=data_monto_melt, x="variable", y="value", hue="ID")
fig1.invert_xaxis()
fig1.set(title='Monto de la compra en los últimos 12 meses para los clientes que se fugan')


data_monto=data.query("fuga == 0")[['ID','monto_1', 'monto_2', 'monto_3', 'monto_4', 'monto_5',
       'monto_6', 'monto_7', 'monto_8', 'monto_9', 'monto_10', 'monto_11',
       'monto_12']][0:3].copy()
data_monto["ID"]=range(data_monto.shape[0])
data_monto_melt=pd.melt(data_monto,id_vars='ID')


fig2=sns.lineplot(data=data_monto_melt, x="variable", y="value", hue="ID")
fig2.invert_xaxis()
fig2.set(title='Monto de la compra en los últimos 12 meses para los clientes que NO se fugan')
