# Importar las bibliotecas necesarias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Cargar el conjunto de datos Iris como ejemplo
data = load_iris()
X = data.data
y = data.target

num_iterations = 10

fig, axs = plt.subplots(2, 5, figsize=(15, 8))

for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}:")
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=iteration)

    # Crear un clasificador KNN con k=3 (puedes ajustar k según tus necesidades)
    knn = KNeighborsClassifier(n_neighbors=5)
    # Entrenar el modelo con los datos de entrenamiento
    knn.fit(X_train, y_train)
    # Realizar predicciones en el conjunto de prueba
    predicciones = knn.predict(X_test)

    print(classification_report(y_test, predicciones))
    cm = confusion_matrix(y_test, predicciones)
    df1 = pd.DataFrame(columns=["0", "1", "2"], index=["0", "1", "2"], data=cm)
    row, col = divmod(iteration, 5)  # Calcular la posición del subplot
    ax = axs[row, col]


    sns.heatmap(df1, annot=True, cmap="Greens", fmt='.0f',
                ax=ax, linewidths=5, cbar=False, annot_kws={"size": 14})
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



