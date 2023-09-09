#----------------------------------------------------------
#
# Date: 06-Sep-2023
#
#           A01752791 Maximiliano Benítez Ahumada
#----------------------------------------------------------

# Importar las bibliotecas necesarias
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap

# Cargar el conjunto de datos Iris como ejemplo
# data = load_iris()
# Cargar el conjunto de datos Digits como ejemplo
data = load_digits()

X, y = data.data, data.target

# Definir el número de iteraciones para evaluar el modelo
num_iterations = 10

# Crear una figura con 2 filas y 5 columnas para los subplots
fig, axs = plt.subplots(2, 5, figsize=(15, 8))

# Iterar a través de las diferentes divisiones de entrenamiento y prueba
for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}:")
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=iteration)

    # Crear un clasificador KNN con k=5 (puedes ajustar k según tus necesidades)
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Entrenar el modelo con los datos de entrenamiento
    knn.fit(X_train, y_train)
    
    # Realizar predicciones en el conjunto de prueba
    predicciones = knn.predict(X_test)

    # Imprimir un informe de clasificación para evaluar el modelo
    print(classification_report(y_test, predicciones))
    
    # Calcular y visualizar la matriz de confusión
    cm = confusion_matrix(y_test, predicciones)

    # Crear un DataFrame de pandas para la matriz de confusión con etiquetas dinámicas
    num_classes = len(np.unique(y_test))
    class_labels = [str(i) for i in range(num_classes)]
    df1 = pd.DataFrame(columns=class_labels, index=class_labels, data=cm)
    
    # Calcular la posición del subplot
    row, col = divmod(iteration, 5)
    ax = axs[row, col]

    sns.heatmap(df1, annot=True, cmap="Greens", fmt='.0f',
                ax=ax, linewidths=5, cbar=False, annot_kws={"size": 12})
    ax.set_xlabel("Predicted Label")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("True Label")
    ax.set_title(f"Iteration {iteration + 1}", size=10)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, predicciones)
    print("Precisión del modelo con Framework:", accuracy)

# Ajustar el diseño de los subplots
plt.tight_layout()
plt.show()




