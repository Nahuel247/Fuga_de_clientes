
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
# Valor_max: Es el valor máx alcanzado por la variable historica que se desea construir

# Si el cliente se va a fugar, se construyen variables hisoticas a partir de una función beta decreciente
# Si el cliente no se va a fugar, se construyen variables historicas a partir de una función normal

# CONSTRUIMOS UNA FUNCIÓN PARA ESTIMAR EL NÚMERO DE ARBOLES OPTIMOS POR MEDIO DE CROSS VALIDATION
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

    accuracy = accuracy_score(
        y_true=y_test,y_pred=predicciones,normalize=True)

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

