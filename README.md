# Detección de obtaculos a partir de datos LiDAR para automoviles autonomos
Trabajo de fin de grado sobre la detección de obstáculos a a partir de datos LiDAR en vehículos autónomos


La detección de obstáculos comienza a ser una necesidad en muchas tareas de automatización.
La visión artificial es uno de los principales métodos utilizados, pero en muchas ocasiones es
insuficiente y necesita de la colaboración de otros sensores.
Este proyecto desarrolla una metodología para la detección de obstáculos en vehículos
autónomos utilizando un escáner laser terrestre, además de su posición se calcula si permanecen
estáticos o si están en movimiento. Con el desplazamiento se calcula si existen posibilidades de
colisión o no con el vehículo no tripulado.
La metodología comienza con la obtención de las nubes obtenidas por el LiDAR para después
reducir su tamaño y posicionarlas en el punto de la trayectoria que corresponde. Se separa el
terreno de los posibles obstáculos para analizar su posición a lo largo de tiempo. Permite
determina la movilidad y los posibles riesgos de colisión.
El análisis del caso de estudio proporciona una idea del funcionamiento de programa,
explicando el rendimiento en cada uno de las diferentes operaciones que se hacen.
Python fue el lenguaje elegido para la programación, y CloudCompare para la visualización de
las nubes, además del análisis de los objetos detectados.

[Detección de obtaculos a partir de datos LiDAR para automoviles autonomos] (https://1drv.ms/b/s!Ag4Ham9-lzVOhdhr0DBVZJQvT5NCVw?e=PhLNwH)
