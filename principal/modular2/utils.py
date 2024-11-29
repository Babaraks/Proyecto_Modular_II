from pymongo import MongoClient
import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import joblib
import pytesseract
import re
import csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_db_credentials():
    dir_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_config = os.path.join(dir_actual,  'config.txt')
    credentials = {}
    with open(ruta_config, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            credentials[key] = value
    return credentials

def get_db_handle():

    credentials = get_db_credentials()

 
    client = MongoClient(
        host=credentials['host'],
        port=int(credentials['port']),
        username=credentials['username'],
        password=credentials['password']
    )
    db_handle = client[credentials['db_name']]
    return db_handle, client

def load_book_images(folder, data_augmentation_enabled=False):
    X_train = []
    y_train = []

    # Obtener la lista de nombres de archivos en la carpeta
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(folder, filename)
            # Cargar la imagen
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Aplicar preprocesamiento para mejorar la detección de texto
                img = cv2.equalizeHist(img)

                # Calcular características (por ejemplo, aspecto y área)
                h, w = img.shape
                aspect_ratio = float(w) / h
                area = h * w
                # Agregar características y etiqueta a X_train y y_train
                X_train.append([filename, aspect_ratio, area])
                y_train.append(1)  # Etiqueta para libros

                if data_augmentation_enabled:
                    # Aplicar aumentos de datos (rotación)
                    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    rotated_aspect_ratio = float(rotated_img.shape[1]) / rotated_img.shape[0]
                    rotated_area = rotated_img.shape[0] * rotated_img.shape[1]
                    X_train.append([filename, rotated_aspect_ratio, rotated_area])
                    y_train.append(1)  # Etiqueta para libros

    # Guardar los datos de entrenamiento en un archivo CSV
    with open('training_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Aspect Ratio', 'Area'])
        writer.writerows(X_train)

    return np.array(X_train), np.array(y_train)


# Función para entrenar y guardar el modelo con ajuste de hiperparámetros
def train_and_save_model(X_train, y_train, model_filename):
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_train[:, 1:], y_train, test_size=0.2, random_state=42)

    # Definir los hiperparámetros a ajustar
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30]
    }

    # Inicializar el clasificador de Random Forest
    classifier = RandomForestClassifier()

    # Realizar la búsqueda de hiperparámetros con validación cruzada
    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo
    best_classifier = grid_search.best_estimator_

    # Evaluar el modelo en el conjunto de prueba
    test_predictions = best_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print("Accuracy on test set:", test_accuracy)

    # Guardar el modelo entrenado
    joblib.dump(best_classifier, model_filename)
    print("Modelo entrenado y guardado.")

    # Guardar los datos de entrenamiento en un archivo CSV
    with open('training_data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(X_train)


# Función para procesar la región del libro y extraer el título del libro
def extract_book_title(book_img):
    # Convertir la imagen a escala de grises
    gray_book_img = cv2.cvtColor(book_img, cv2.COLOR_BGR2GRAY)

    # Aplicar ecualización del histograma para mejorar el contraste
    equalized_img = cv2.equalizeHist(gray_book_img)

    # Utilizar OCR para extraer el texto del libro
    custom_config = r'--oem 3 --psm 6'  # Configuración personalizada para mejorar el OCR
    book_title = pytesseract.image_to_string(equalized_img, config=custom_config)

    # Limpiar el texto extraído
    cleaned_title = re.sub(r'[^\w\s]', '', book_title)  # Eliminar símbolos y puntuación
    cleaned_title = re.sub(r'\b\w{1}\b', '', cleaned_title)  # Eliminar palabras de una sola letra

    # Eliminar espacios adicionales y convertir a mayúsculas
    cleaned_title = ' '.join(cleaned_title.split()).upper()

    return cleaned_title


def load_and_preprocess_images(images):
    processed_images = []
    for image in images:
        try:
            # Convertir la imagen a escala de grises si no lo está
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Redimensionar la imagen a un tamaño fijo (opcional)
            resized_image = cv2.resize(gray_image, (250, 250))

            # Aplanar la imagen y agregarla a la lista de imágenes procesadas
            processed_images.append(resized_image.flatten())
        except Exception as e:
            print(f"Error al procesar la imagen: {str(e)}")

    return np.array(processed_images)

def documento_a_nparray(documento):

    coordenadas = documento['caracteristicas']
    return np.array(coordenadas)