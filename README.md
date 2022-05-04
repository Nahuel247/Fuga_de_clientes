# Fuga de clientes (I parete: Desarrollo del modelo)

La fuga de clientes es una de las principales problemáticas que puede tener una empresa, tema que debe ser tratado con suma atención ya que ganar un cliente es 7 veces más difícil que retenerlo.\
\
En este repositorio encontraran el desarrollo metodológico para identificar, a través de Machine Learning, aquellos clientes más propensos a fugarse en el próximo mes (información relevante para realizar retención), en particular:

* Se utilizó el modelo RandomForest, una metodología que se basa en árboles de decisiones.
* Se construyó una base de datos con 1000 registros y una tasa de fuga del 15%.
* Los datos se construyeron a partir de supuestos plausibles sobre el comportamiento de los clientes en los 12 meses anteriores al mes de referencia, condicionado sobre si se fugará o no el próximo mes.
* Las variables históricas para los clientes que se fugaban seguían una distribución beta decreciente y los que no una distribución normal, en ambos casos se agregó ruido a través de una distribución uniforme.
* Las variables que se construyeron fueron: total de la compra mensual, nivel de satisfacción, tiempo de espera, diversidad de la canasta, antigüedad del cliente, tipo de cliente.
* El modelo fue construido a través de una muestra de construcción, validado mediante cross-validation y testeado en una muestra out of sample.
* El desempeño del modelo fue estimado a través de indicadores como el ginni y accuracy.

#Algunos resultados

# Comportamiento de compra de los clientes que no se fugan
[![Monto-total-de-la-compra-clientes-que-NO-se-fugan.png](https://i.postimg.cc/yxpcTksh/Monto-total-de-la-compra-clientes-que-NO-se-fugan.png)](https://postimg.cc/4HtYf4Kn)

# Comportamiento de compra de los clientes que se fugan 
[![Monto-total-de-la-compra-clientes-que-se-fugan.png](https://i.postimg.cc/nV34xwBs/Monto-total-de-la-compra-clientes-que-se-fugan.png)](https://postimg.cc/w3syc2WH)

# Cross validation
[![cross-validation.png](https://i.postimg.cc/GpfRrKMk/cross-validation.png)](https://postimg.cc/kVWzvQP4)

# Accuracy Cross validation vs Train
[![Accuracy-CV.png](https://i.postimg.cc/9f02vKBm/Accuracy-CV.png)](https://postimg.cc/DS9RGCDH)


# Tabla de eficiencia
[![Tabla-de-eficiencia.png](https://i.postimg.cc/nhF5SW43/Tabla-de-eficiencia.png)](https://postimg.cc/mtn8t869)

# SIGUIENTES ETAPAS
* Desarrollo de un análisis al modelo y definición de gestiones
* Desarrollo de un Dashbord para su ejecución mensual
