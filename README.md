# Momento de Retroalimentación: Módulo 2 Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución. (Portafolio Implementación)

## Maximiliano Benítez Ahumada - A01752791

Para la realización de esta actividad se implementó el algoritmo KNN para clasificación.

# ¿Cómo funciona KNN?

K-Nearest Neighbors (k-NN): Algoritmo de aprendizaje supervisado para clasificación y regresión.

1. Entrenamiento: Almacena ejemplos con etiquetas conocidas.
2. Predicción (Clasificación): Encuentra los "k" ejemplos más cercanos en el conjunto de entrenamiento y asigna la etiqueta más común.
3. Predicción (Regresión): Encuentra los "k" ejemplos más cercanos y promedia sus valores para predecir un número.
4. El valor de "k" afecta la sensibilidad del algoritmo. Simple pero puede ser sensible al ruido y a la elección de la métrica de distancia.

# Manual de Instalación y Configuración de Miniconda y Entorno Virtual con scikit-learn para el uso del algoritmo KNN.

En este manua se presentan los pasos para instalar Miniconda, una versión minimalista de Anaconda, y cómo crear un entorno virtual utilizando Miniconda para trabajar con scikit-learn, una biblioteca de aprendizaje automático en Python con la cual se implementó KNN.

## Paso 1: Descargar Miniconda

1. Accede al sitio web de Miniconda: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

2. Selecciona la versión de Miniconda adecuada para tu sistema operativo (Windows, macOS o Linux).

3. Descarga el instalador y sigue las instrucciones de instalación según tu sistema operativo.

## Paso 2: Instalar Miniconda

### Windows

1. Ejecuta el instalador descargado.

2. Acepta los términos y condiciones.

3. Elige la opción "Instalar solo para mí" y selecciona una ubicación para la instalación (por defecto, `C:\Users\<TuUsuario>\Miniconda3`).

4. Selecciona "Agregar Anaconda a mi PATH" durante la instalación.

5. Haz clic en "Instalar".

### macOS y Linux

1. Abre una terminal.

2. Navega hacia la ubicación del instalador descargado (usando el comando `cd`).

3. Ejecuta el comando de instalación: 

   ```
   bash Miniconda3-latest-MacOSX-x86_64.sh
   ```

   (El nombre del archivo puede variar según la versión descargada).

4. Acepta los términos y condiciones.

5. Sigue las instrucciones en pantalla y proporciona una ubicación para la instalación (por defecto, en tu directorio de inicio).

6. Acepta agregar el inicio de Miniconda al archivo `.bashrc` o `.zshrc` si se te pregunta.

## Paso 3: Crear un entorno virtual con scikit-learn

1. Abre una nueva terminal (si no tienes una abierta).

2. Crea un nuevo entorno virtual con el nombre "sklearn-env" y la versión de Python deseada (por ejemplo, Python 3.8):

   ```
   conda create -n sklearn-env python=3.8
   ```

3. Activa el entorno virtual:

   - En Windows:

     ```
     conda activate sklearn-env
     ```

   - En macOS y Linux:

     ```
     source activate sklearn-env
     ```

4. Instala scikit-learn en el entorno virtual:

   ```
   conda install scikit-learn
   ```

5. ¡Listo! Ahora tienes un entorno virtual con Miniconda y scikit-learn instalados.

## Paso 4: Usar el entorno virtual

1. Cada vez que quieras trabajar con scikit-learn, activa el entorno virtual:

   - En Windows:

     ```
     conda activate sklearn-env
     ```

   - En macOS y Linux:

     ```
     source activate sklearn-env
     ```
2. Posteriormente se podrá ejecutar el archivo 'train.py', el cual ejecuta todas las funciones necesarias de predicción.

## Output

El output de 'train.py' despliega una lista de los datos que el modelo predijo, así como la gráfica y el porcentaje de "accuracy" de los mismos.


     
     [7, 1, 7, 6, 0, 2, 4, 3, 6, 3, 7, 8, 7, 9, 4, 3, 8, 7, 8, 4, 0, 3, 9, 1, 3, 6, 6, 0, 5, 4, 1, 8, 1, 2, 3, 2, 7, 6, 4, 8, 6, 4, 4, 0, 9, 2, 8, 5, 4, 4, 4, 1, 7, 6, 8, 2, 9,         9, 9, 0, 8, 3, 1, 8, 8, 1, 3, 9, 1, 3, 9, 6, 9, 5, 2, 1, 9, 2, 1, 3, 8, 7, 3, 3, 8, 7, 7, 5, 8, 2, 6, 1, 9, 1, 6, 4, 5, 2, 2, 4, 5, 4, 7, 6, 5, 9, 2, 4, 1, 0, 7, 6, 1, 2,         9, 5, 2, 5, 0, 3, 2, 7, 6, 4, 8, 2, 1, 1, 6, 4, 6, 2, 5, 4, 7, 5, 0, 9, 1, 0, 5, 6, 7, 6, 3, 8, 3, 2, 0, 4, 4, 1, 5, 4, 6, 1, 1, 1, 6, 1, 7, 9, 0, 7, 9, 5, 4, 1, 3, 8, 6,         4, 7, 1, 5, 7, 4, 7, 4, 3, 2, 2, 1, 1, 4, 4, 3, 5, 5, 9, 4, 5, 5, 9, 3, 9, 3, 1, 2, 0, 8, 2, 4, 5, 2, 4, 6, 8, 3, 8, 1, 0, 8, 1, 8, 5, 6, 8, 7, 1, 8, 3, 4, 9, 7, 0, 5, 5,         6, 1, 3, 0, 5, 8, 2, 0, 9, 8, 6, 7, 8, 4, 1, 0, 5, 2, 5, 1, 6, 4, 7, 1, 2, 6, 4, 4, 6, 3, 2, 3, 2, 6, 5, 2, 9, 4, 7, 0, 1, 0, 4, 3, 1, 2, 7, 9, 8, 5, 9, 5, 7, 0, 4, 8, 4,         9, 4, 0, 7, 7, 7, 5, 3, 5, 3, 9, 7, 5, 5, 2, 7, 0, 8, 9, 1, 7, 9, 8, 5, 0, 2, 0, 8, 7, 0, 9, 5, 5, 9, 6, 1, 2, 3, 9, 6, 3, 2, 9, 3, 4, 3, 4, 1, 8, 1, 8, 5, 0, 9, 2, 7, 2,         3, 5, 2, 6, 3, 4, 1, 5, 0, 5, 4, 6, 3, 2, 5, 0, 7, 3]
     
      0.9861111111111112

    

## Conclusión
1. Cuando haya terminado de trabajar, se debe desactiva el entorno virtual:

   ```
   conda deactivate
   ```


¡Se ha completado la creación de un entorno virtual, la instalación de scikit-learn y el testeo del algoritmo KNN! ¡Happy Hacking! 🚀


