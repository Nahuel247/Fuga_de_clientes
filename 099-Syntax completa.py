
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
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('once')


#############################################
# CREAMOS LAS FUNCIONES QUE VAMOS A UTILIZAR
#############################################

# Construiremos una función que defina el comportamiento historico del cliente según si su marca de fuga es 0 o 1

# Rezagos: Número de meses anteriores al mes de referencia con información, 1 es el mes más cercano y 12 es el más lejano
# Valor_max: Es un valor que regula el valor máximo alcanzado por la variable historica que se desea construir

# Si el cliente se va a fugar, se construyen variables hisoticas a partir de una función beta decreciente
# Si el cliente no se va a fugar, se construyen variables historicas a partir de una función normal

# Creamos una función que defina la tendencia que tomará la variable historica según si el cliente se fuga o no
def tendencia(x, rezagos,valor_max):
    ds=valor_max/100# ds
    noise = np.random.uniform(-1, 1, rezagos)  # Se utilizará para agregar ruido
    if x==1: # Distribución beta (el cliente se fuga, "fuga" ==1)
        a, b, inicio, fin = 10, 2, 0.1, 0.99  # párametros de la distribución beta
        x = np.linspace(beta.ppf(inicio, a, b), beta.ppf(fin, a, b), 12)
        x_valores = beta.pdf(x, a, b)
        x_valores= (x_valores/np.max(x_valores)) * valor_max * rand(1)
        x_valores = x_valores + (x_valores * noise)
    else: # Distribución normal (el cliente NO se fuga, "fuga" ==0)
        x_valores = np.random.normal(valor_max*rand(1), ds * rand(1), rezagos)
        x_valores = x_valores + (x_valores * noise)
        x_valores = (x_valores/np.max(x_valores)) * valor_max * rand(1)
    return pd.Series(x_valores)


# CONSTRUIMOS UNA FUNCIÓN PARA ESTIMAR EL NÚMERO DE ARBOLES OPTIMOS POR MEDIO DE CROSS VALIDATION
# CONSTRUIMOS UNA FUNCIÓN PARA ESTIMAR EL DESEMPEÑO DEL MODELO A TRAVÉS DE CROSS VALIDATION Y
# EL NÚMERO DE ARBOLES OPTIMOS

def modelo_cv(X_train,y_train,estimator_range,parametros):
    train_scores = []
    cv_scores = []

    for n_estimators in estimator_range:
        modelo=RandomForestClassifier(
            n_estimators=n_estimators,
            **parametros)

        # Error de train
        modelo.fit(X_train, y_train)
        predicciones = modelo.predict(X=X_train)

        #rmse = mean_squared_error(y_true=y_train,y_pred=predicciones,squared=False) # para var continua
        acc=accuracy_score(y_true=y_train, y_pred=predicciones,normalize=True)
        train_scores.append(acc)

        # Error de validación cruzada
        scores = cross_val_score(
            estimator=modelo,
            X=X_train,
            y=y_train,
            #scoring='neg_root_mean_squared_error', # para variable continua
            scoring='accuracy',# para (2) variables categoricas
            cv=5)
        cv_scores.append(scores.mean())
    return train_scores, cv_scores

# FUNCIÓN PARA GRAFICAR EL AJUSTE DEL MODELO BAJO DISTINTOS NÚMERO DE ÁRBOLES
def grafico_ajuste(estimator_range,train_scores,cv_scores):
    fig, ax = plt.subplots(figsize=(6, 3.84))
    ax.plot(estimator_range, train_scores, label="train scores")
    ax.plot(estimator_range, cv_scores, label="cv scores")
    ax.plot(estimator_range[np.argmax(cv_scores)], max(cv_scores),
            marker='o', color="red", label="max score")
    #ax.set_ylabel("root_mean_squared_error")
    ax.set_ylabel("accuracy")
    ax.set_xlabel("n_estimators")
    ax.set_title("Evolución del cv-accuracy vs número de árboles")
    plt.legend();
    print(f"Valor óptimo de n_estimators: {estimator_range[np.argmax(cv_scores)]}")


# SE CREA UNA FUNCIÓN PARA EVALUAR DISTINTOS INDICADORES
def tablas_eficiencia(y_test,predicciones):
    mat_confusion = confusion_matrix(y_true=y_test,y_pred=predicciones)

    accuracy = accuracy_score(y_true=y_test,y_pred=predicciones,normalize=True)

    print("Matriz de confusión")
    print("-------------------")
    print(mat_confusion)
    print("")
    print(f"El accuracy de test es: {100 * accuracy} %")

    print(classification_report(
            y_true=y_test,
            y_pred=predicciones))

# SE CONSTRUYE LA FUNCIÓN DE GINI
def gini_generico(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini(actual, pred):
    return gini_generico(actual, pred) / gini_generico(actual, actual)


###########################################################################
# CREAMOS LA BASE DE DATOS SEGÚN TENDENCIAS PLAUSIBLES ASOCIADAS A LA FUGA
###########################################################################

n=1000 # Número de clientes/registros que se van a crear
rezagos=12 # Número de meses anteriores al mes de referencia para los cuales se crearán variables

#Inicializamos un dataframe con los ID de cada cliente
data=pd.DataFrame({"ID": [uuid.uuid4() for _ in range(n)]})

# Agregamos una tasa de fuga del 15% (FUGA: 1: NO compra el siguiente mes, 0: SI lo hace)
data=data.assign(fuga=random.choices([1,0], cum_weights=(15,85),k = n))


##### VARIABLES "CAPTURADAS" POR la empresa que ofrece sus servicios ######

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


#######################################
# CREACIÓN DE VARIABLES ARTIFICIALES
#######################################

# Construimos una base de datos que entrará en el modelo
data_artificial=data[["fuga","antiguedad","grande","pequeno"]]

# truquito para crear las variables de forma aútomatica
# Truquito para crear las variables de forma aútomatica: A continuación se crean de forma automatica la programación
# de múltiples variables, están quedan guardads en el objeto "variables" y pueden ser ejecutadas con exec o impresas
# para dejarlas explicitas

list_var=["monto_","satisfaccion_","canasta_","espera_"]
meses=[12,6,3]
funciones=["sum","mean","max","min"]

#guardamos en "variables" las variables creadas, al imprimirlo se puede ver el listado de variables creadas
variables=[("data_artificial=data_artificial.assign("+var+funcion+str(n_meses)+"=data[['"+var+"' +str(i+1) for i in range("+str(n_meses)+")]]."+funcion+"(axis=1))") for var in list_var for n_meses in meses for funcion in funciones]

# ejecutamos las funciones
#for var in variables:
#    exec(var)

# dejamos explicitas las funciones creadas

# monto total, max, min, promedio de los últimos 12, 6 y 3 meses para cada cliente
data_artificial=data_artificial.assign(monto_sum12=data[['monto_' +str(i+1) for i in range(12)]].sum(axis=1))
data_artificial=data_artificial.assign(monto_mean12=data[['monto_' +str(i+1) for i in range(12)]].mean(axis=1))
data_artificial=data_artificial.assign(monto_max12=data[['monto_' +str(i+1) for i in range(12)]].max(axis=1))
data_artificial=data_artificial.assign(monto_min12=data[['monto_' +str(i+1) for i in range(12)]].min(axis=1))
data_artificial=data_artificial.assign(monto_sum6=data[['monto_' +str(i+1) for i in range(6)]].sum(axis=1))
data_artificial=data_artificial.assign(monto_mean6=data[['monto_' +str(i+1) for i in range(6)]].mean(axis=1))
data_artificial=data_artificial.assign(monto_max6=data[['monto_' +str(i+1) for i in range(6)]].max(axis=1))
data_artificial=data_artificial.assign(monto_min6=data[['monto_' +str(i+1) for i in range(6)]].min(axis=1))
data_artificial=data_artificial.assign(monto_sum3=data[['monto_' +str(i+1) for i in range(3)]].sum(axis=1))
data_artificial=data_artificial.assign(monto_mean3=data[['monto_' +str(i+1) for i in range(3)]].mean(axis=1))
data_artificial=data_artificial.assign(monto_max3=data[['monto_' +str(i+1) for i in range(3)]].max(axis=1))
data_artificial=data_artificial.assign(monto_min3=data[['monto_' +str(i+1) for i in range(3)]].min(axis=1))

# indicador de satisfacción, total, max, min, promedio de los últimos 12, 6 y 3 meses para cada cliente
data_artificial=data_artificial.assign(satisfaccion_sum12=data[['satisfaccion_' +str(i+1) for i in range(12)]].sum(axis=1))
data_artificial=data_artificial.assign(satisfaccion_mean12=data[['satisfaccion_' +str(i+1) for i in range(12)]].mean(axis=1))
data_artificial=data_artificial.assign(satisfaccion_max12=data[['satisfaccion_' +str(i+1) for i in range(12)]].max(axis=1))
data_artificial=data_artificial.assign(satisfaccion_min12=data[['satisfaccion_' +str(i+1) for i in range(12)]].min(axis=1))
data_artificial=data_artificial.assign(satisfaccion_sum6=data[['satisfaccion_' +str(i+1) for i in range(6)]].sum(axis=1))
data_artificial=data_artificial.assign(satisfaccion_mean6=data[['satisfaccion_' +str(i+1) for i in range(6)]].mean(axis=1))
data_artificial=data_artificial.assign(satisfaccion_max6=data[['satisfaccion_' +str(i+1) for i in range(6)]].max(axis=1))
data_artificial=data_artificial.assign(satisfaccion_min6=data[['satisfaccion_' +str(i+1) for i in range(6)]].min(axis=1))
data_artificial=data_artificial.assign(satisfaccion_sum3=data[['satisfaccion_' +str(i+1) for i in range(3)]].sum(axis=1))
data_artificial=data_artificial.assign(satisfaccion_mean3=data[['satisfaccion_' +str(i+1) for i in range(3)]].mean(axis=1))
data_artificial=data_artificial.assign(satisfaccion_max3=data[['satisfaccion_' +str(i+1) for i in range(3)]].max(axis=1))
data_artificial=data_artificial.assign(satisfaccion_min3=data[['satisfaccion_' +str(i+1) for i in range(3)]].min(axis=1))

# diversificación de la canasta total, max, min, promedio de los últimos 12, 6 y 3 meses para cada cliente
data_artificial=data_artificial.assign(canasta_sum12=data[['canasta_' +str(i+1) for i in range(12)]].sum(axis=1))
data_artificial=data_artificial.assign(canasta_mean12=data[['canasta_' +str(i+1) for i in range(12)]].mean(axis=1))
data_artificial=data_artificial.assign(canasta_max12=data[['canasta_' +str(i+1) for i in range(12)]].max(axis=1))
data_artificial=data_artificial.assign(canasta_min12=data[['canasta_' +str(i+1) for i in range(12)]].min(axis=1))
data_artificial=data_artificial.assign(canasta_sum6=data[['canasta_' +str(i+1) for i in range(6)]].sum(axis=1))
data_artificial=data_artificial.assign(canasta_mean6=data[['canasta_' +str(i+1) for i in range(6)]].mean(axis=1))
data_artificial=data_artificial.assign(canasta_max6=data[['canasta_' +str(i+1) for i in range(6)]].max(axis=1))
data_artificial=data_artificial.assign(canasta_min6=data[['canasta_' +str(i+1) for i in range(6)]].min(axis=1))
data_artificial=data_artificial.assign(canasta_sum3=data[['canasta_' +str(i+1) for i in range(3)]].sum(axis=1))
data_artificial=data_artificial.assign(canasta_mean3=data[['canasta_' +str(i+1) for i in range(3)]].mean(axis=1))
data_artificial=data_artificial.assign(canasta_max3=data[['canasta_' +str(i+1) for i in range(3)]].max(axis=1))
data_artificial=data_artificial.assign(canasta_min3=data[['canasta_' +str(i+1) for i in range(3)]].min(axis=1))


# tiempo de espera total, max, min, promedio de los últimos 12, 6 y 3 meses para cada cliente
data_artificial=data_artificial.assign(espera_sum12=data[['espera_' +str(i+1) for i in range(12)]].sum(axis=1))
data_artificial=data_artificial.assign(espera_mean12=data[['espera_' +str(i+1) for i in range(12)]].mean(axis=1))
data_artificial=data_artificial.assign(espera_max12=data[['espera_' +str(i+1) for i in range(12)]].max(axis=1))
data_artificial=data_artificial.assign(espera_min12=data[['espera_' +str(i+1) for i in range(12)]].min(axis=1))
data_artificial=data_artificial.assign(espera_sum6=data[['espera_' +str(i+1) for i in range(6)]].sum(axis=1))
data_artificial=data_artificial.assign(espera_mean6=data[['espera_' +str(i+1) for i in range(6)]].mean(axis=1))
data_artificial=data_artificial.assign(espera_max6=data[['espera_' +str(i+1) for i in range(6)]].max(axis=1))
data_artificial=data_artificial.assign(espera_min6=data[['espera_' +str(i+1) for i in range(6)]].min(axis=1))
data_artificial=data_artificial.assign(espera_sum3=data[['espera_' +str(i+1) for i in range(3)]].sum(axis=1))
data_artificial=data_artificial.assign(espera_mean3=data[['espera_' +str(i+1) for i in range(3)]].mean(axis=1))
data_artificial=data_artificial.assign(espera_max3=data[['espera_' +str(i+1) for i in range(3)]].max(axis=1))
data_artificial=data_artificial.assign(espera_min3=data[['espera_' +str(i+1) for i in range(3)]].min(axis=1))


##########################################
# CONSTRUIMOS EL MODELO
##########################################

# División de los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(
                                        data_artificial.drop(columns = "fuga"),
                                        data_artificial["fuga"],
                                        random_state = 123
                                    )

# Definimos los párametros para el modelo
parametros=({
        "criterion":"entropy",
        "max_depth":3,
        "max_features":0.8,
        "oob_score": False,
        "n_jobs":-1,
        "random_state":123})

# Entrenamos modelos con n_estimadores (árboles) mediante CV
estimator_range = range(1, 150, 5)
train_scores, cv_scores=modelo_cv(X_train,y_train,estimator_range,parametros)

# Gráfico para visualizar el ajuste del modelo y el número de árboles optimos (n_estimators)
grafico_ajuste(estimator_range,train_scores,cv_scores)

modelo = RandomForestClassifier(
            n_estimators = 6, # para obtener resultados realistas, se le da un valor diferente al recomendado
            **parametros)

# Entrenamiento del modelo
modelo.fit(X_train, y_train)

# Predicción del modelo
predicciones = modelo.predict(X = X_test)


#################
#   DESEMPEÑO
#################

tablas_eficiencia(y_test,predicciones)

predicciones_prob = pd.DataFrame(modelo.predict_proba(X = X_test))[1]
gini(y_test,predicciones_prob)


###################
#   REFERENCIA
###################

#Teoría:
# https://www.cienciadedatos.net/documentos/py08_random_forest_python.html
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://www.kaggle.com/code/batzner/gini-coefficient-an-intuitive-explanation/notebook


#Fuga de clientes:
#https://www.questionpro.com/blog/es/fuga-de-clientes/
#https://findialeyva.com/tasas-de-abandono-y-fuga-de-clientes/
#http://repobib.ubiobio.cl/jspui/bitstream/123456789/2308/1/Ovalle_Retamal_Victor_Francisco_Javier.pdf
#http://opac.pucv.cl/pucv_txt/txt-0000/UCF0318_01.pdf
#https://hexa-soluciones.es/reduccion-de-fuga-de-clientes-analisis/
