En la carpeta Release está el make generado por eclipse, con los flags de optimización que pone (tipo O3...)

He dejado varias implementaciones:
    - Una versión con memoria compartida y múltiples kernels.
        Ejecutar con el parámetro 'm'
    - Otra que ejecuta todo en un único kernel, utilizando muchos registros y memoria compartida (el profiler avisa de
    que para mi gráfica, 660gtx, usa demasiados kernels por bloque/thread, y que por tanto no puede lanzar todos los
    bloques a ejecución); y tiene problemas similares con la memoria compartida, que usa demasiada.
        Aún así, es más rápida que la versión anterior. 
        He realizado múltiples compilaciones para ver que parámetros (tamaño de bloque...) eran óptimos.
        Ejecutar con 'g'

    Optimizaciones comunes:
        - Se utiliza memoria no pageable en todas las implementaciones
        - Para el kernel monolítico, se utiliza memoria de constantes. No me dio tiempo a ahcerlo para la otra versión,
        pero vamos que tampoco había mucha diferencia.

    También hay otros modos de ejecución:
        'b' para comprobar los tiempos de ejecución de todas las versiones (incluida la gpu)
        'a' para realizar 10 pasadas sobre la versión coalescente y sacar la media de tiempo de ejecución.
        'f' para comprobar si la salida es correcta (comparando con cpu)

    Fallos conocidos:
        - Las versiones de un solo kernel no computan bien los bordes de la imagen (5 por lado)
        - Para algunas imágenes (ninguna de las que nos dejaste) diferái de la CPU en algunos píxeles (en concreto, para una
        images de 200M pixeles fallaba en 45, para otra de 400M solo en uno). Esto para todas las versiones.

    Trabajo por hacer:
        - Las implementaciones de un kernel creo que les queda algo de espacio para mejora, tienen algo de memoria
        compartida desaprovechada (solo se utiliza toda la matriz reservada en la carga de la imagen, luego se dejan
        algunos bordes sin usar), la carga de memoria, tanto como global y compartida, suele dejar threads inactivos,
        habría que ver si se pueden mejorar esto sin alterar el rendimiento por accesos desalineados.
		- Tenía una versión para asegurar el acceso coalescente, pero justo después de enviarla me di cuenta de que etsaba 
		mal y tuve que reenviar. De todas formas será un fallo tonto que no copia bien la memoria. En cuanto a tiempo de ejecución,
		para una imagen muy grande solo ganaba un milisegudndo (0,5% de incremento creo recordar) así que no era mucho.
        - Quedaría por mirar el uso de streams, pero creo que es bastante complejo (por el tema de cargar submatrices
        con los halso correspondientes). Aún así, para el kernel único sólo hay dos transiciones de memoria global, así
        que posiblemente no se ganaría demasiado.
