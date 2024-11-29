from django.shortcuts import render
from django.shortcuts import redirect
from django.http import HttpResponse
from django.template import loader
from django.contrib import messages
from .utils import get_db_handle,train_and_save_model,extract_book_title,load_and_preprocess_images
from datetime import date
from django.http import HttpResponse
from django.shortcuts import render
from django.http import HttpResponse
import joblib
import os
import numpy as np
import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from django.urls import reverse
from scipy.spatial import distance
import re
from django.contrib import messages

classifier = ""
img_counter=0
data_augmentation_enabled = True
nombre_user=""
bander=False
def login(request):
    global nombre_user
    global bander
    if request.method == 'POST':
        usuario = request.POST.get('usuario')
        contraseña = request.POST.get('contraseña')
        db_handle, client = get_db_handle()

        # Buscar en la colección de administradores
        admin_user = db_handle.admin.find_one({'usuario': usuario, 'contraseña': contraseña})
        if admin_user:
            client.close()
            nombre_user = usuario
            bander = True
            request.session['bander'] = bander  # Guardar bandera en la sesión
            return redirect('lista_libro')

        # Buscar en la colección de encargados
        encargado_user = db_handle.encargado.find_one({'usuario': usuario, 'contraseña': contraseña})
        if encargado_user:
            nombre_user = usuario
            client.close()
            bander = False
            request.session['bander'] = bander  # Guardar bandera en la sesión
            return redirect('lista_libro')

        messages.error(request, 'Credenciales inválidas')
        client.close()
        return redirect('Login')

    return render(request, 'login.html')


        

def panel_encargado(request):

    return render(request,'encargado_panel.html')    

def leer_encargado(request):
    db_handle, client = get_db_handle()
    encargados = db_handle.encargado.find()
    html_template = loader.get_template('detalles_encargado.html')
    rendered_html = html_template.render({'encargados': encargados}, request)
    client.close()  
    return HttpResponse(rendered_html)

def actualizar_encargado(request, id_encargado):
    if request.method == 'POST':
        # Obtiene los datos actualizados del formulario
        id_encargado = int(request.POST['id_encargado'])
        nombre = request.POST['nombre']
        apellido_p = request.POST['apellido_p']
        apellido_m = request.POST['apellido_m']
        usuario = request.POST['usuario']
        contraseña = request.POST['contraseña']

        # Actualiza el documento en la colección "encargado"
        db_handle, client = get_db_handle()
        db_handle.encargado.update_one({"id_encargado": id_encargado}, {"$set": {
            "nombre": nombre,
            "apellido_p": apellido_p,
            "apellido_m": apellido_m,
            "usuario": usuario,
            "contraseña": contraseña
        }})
        client.close()
        return redirect('lista_encargado')
    else:
        # Muestra el formulario de actualización
        db_handle, client = get_db_handle()
        encargado = db_handle.encargado.find_one({"id_encargado": id_encargado})
        client.close()
        return render(request, 'formulario_actualizar.html', {'encargado': encargado})
    
def eliminar_encargado( request, id_encargado):
    # Elimina el encargado de la base de datos utilizando su id_encargado
    db_handle, client = get_db_handle()
    db_handle.encargado.delete_many({"id_encargado": id_encargado})

    client.close()
    
    # Redirige al usuario a la lista de encargados después de eliminar exitosamente
    return redirect('lista_encargado')

def obtener_ultimo_id_encargado(db_handle):
    ultimo_encargado = db_handle.encargado.find_one({}, sort=[("id_encargado", -1)])
    if ultimo_encargado:
        return int(ultimo_encargado['id_encargado']) + 1
    else:
        return 1

def crear_encargado(request):
    if request.method == 'POST':
        # Obtén los datos del formulario
        nombre = request.POST['nombres']
        apellido_p = request.POST['apellido_p']
        apellido_m = request.POST['apellido_m']
        usuario = request.POST['usuario']
        contraseña = request.POST['contraseña']
        
        # Obtiene el último id_encargado y genera el siguiente ID
        db_handle, client = get_db_handle()
        id_encargado = obtener_ultimo_id_encargado(db_handle)
        
        # Crea un nuevo documento de encargado
        encargado = {
            "id_encargado": int(id_encargado),  
            "nombre": nombre,
            "apellido_p": apellido_p,
            "apellido_m": apellido_m,
            "usuario": usuario,
            "contraseña": contraseña
        }
        
        # Guarda el documento en la base de datos
        db_handle.encargado.insert_one(encargado)
        client.close()
        
        return redirect('lista_encargado')
    else:
        return render(request, 'formulario_crear.html')




def leer_admin(request):
    db_handle, client = get_db_handle()
    admins = db_handle.admin.find()
    html_template = loader.get_template('detalles_admin.html')
    rendered_html = html_template.render({'admins': admins}, request)
    client.close()  
    return HttpResponse(rendered_html)

def actualizar_admin(request, id_admin):
    if request.method == 'POST':
        # Obtiene los datos actualizados del formulario
        id_admin = int(request.POST['id_admin'])
        nombre = request.POST['nombre']
        apellido_p = request.POST['apellido_p']
        apellido_m = request.POST['apellido_m']
        correo = request.POST['correo']
        usuario = request.POST['usuario']
        contraseña = request.POST['contraseña']

        # Actualiza el documento en la colección "admin"
        db_handle, client = get_db_handle()
        db_handle.admin.update_one({"id_admin": id_admin}, {"$set": {
            "nombre": nombre,
            "apellido_p": apellido_p,
            "apellido_m": apellido_m,
            "correo" : correo,
            "usuario": usuario,
            "contraseña": contraseña
        }})
        client.close()
        return redirect('lista_admin')
    else:
        # Muestra el formulario de actualización
        db_handle, client = get_db_handle()
        admin = db_handle.admin.find_one({"id_admin": id_admin})
        client.close()
        return render(request, 'formulario_actualiza_A.html', {'admin': admin})
    
def eliminar_admin( request,id_admin):
    # Elimina el admin de la base de datos utilizando su id_admin
    db_handle, client = get_db_handle()
    db_handle.admin.delete_many({"id_admin": id_admin})

    client.close()
    
    # Redirige al admin a la lista de admins después de eliminar exitosamente
    return redirect('lista_admin')

def obtener_ultimo_id_admin(db_handle):
    ultimo_admin = db_handle.admin.find_one({}, sort=[("id_admin", -1)])
    if ultimo_admin:
        return int(ultimo_admin['id_admin']) + 1
    else:
        return 1

def crear_admin(request):
    if request.method == 'POST':
        # Obtén los datos del formulario
        nombre = request.POST['nombre']
        apellido_p = request.POST['apellido_p']
        apellido_m = request.POST['apellido_m']
        correo = request.POST['correo']
        usuario = request.POST['usuario']
        contraseña = request.POST['contraseña']
        
        # Obtiene el último id_admin y genera el siguiente ID
        db_handle, client = get_db_handle()
        id_admin = obtener_ultimo_id_admin(db_handle)
        
        # Crea un nuevo documento de admin
        admin = {
            "id_admin": int(id_admin),  
            "nombre": nombre,
            "apellido_p": apellido_p,
            "apellido_m": apellido_m,
            "correo" : correo,
            "usuario": usuario,
            "contraseña": contraseña
        }
        
        # Guarda el documento en la base de datos
        db_handle.admin.insert_one(admin)
        client.close()
        
        return redirect('lista_admin')
    else:
        return render(request, 'formulario_crear_A.html')

def panel_admin(request):
    global bander
    if bander == True:
        return render(request,'Admin_panel.html')
    elif bander == False:
        return render(request,'encargado_panel.html')




def leer_usuario(request):
    db_handle, client = get_db_handle()
    usuarios = db_handle.usuarios.find()
    html_template = loader.get_template('detalles_usuario.html')
    rendered_html = html_template.render({'usuarios': usuarios}, request)
    client.close()  
    return HttpResponse(rendered_html)

def actualizar_usuario(request, id_usuario):
    if request.method == 'POST':
        # Obtiene los datos actualizados del formulario
        id_usuario = int(request.POST['id_usuario'])
        nombre = request.POST['nombre']
        apellido_p = request.POST['apellido_p']
        apellido_m = request.POST['apellido_m']
        matricula = request.POST['matricula']
        carrera = request.POST['carrera']
        grado = request.POST['grado']
        contraseña = request.POST['contraseña']

        # Actualiza el documento en la colección "usuario"
        db_handle, client = get_db_handle()
        db_handle.usuarios.update_one({"id_usuario": id_usuario}, {"$set": {
            "nombre": nombre,
            "apellido_p": apellido_p,
            "apellido_m": apellido_m,
            "matricula" : matricula,
            "carrera" : carrera,
            "grado" : grado,
            "contraseña": contraseña
        }})
        client.close()
        return redirect('lista_usuario')
    else:
        # Muestra el formulario de actualización
        db_handle, client = get_db_handle()
        usuario = db_handle.usuarios.find_one({"id_usuario": id_usuario})
        client.close()
        return render(request, 'formulario_actualiza_U.html', {'usuario': usuario})
    
def eliminar_usuario(request, id_usuario):
    # Elimina el usuario de la base de datos utilizando su id_usuario
    db_handle, client = get_db_handle()
    db_handle.usuarios.delete_one({"id_usuario": id_usuario})

    client.close()
    
    # Redirige al usuario a la lista de usuarios después de eliminar exitosamente
    return redirect('lista_usuario')
    
def obtener_ultimo_id_usuario(db_handle):
    ultimo_usuario = db_handle.usuarios.find_one({}, sort=[("id_usuario", -1)])
    if ultimo_usuario:
        return int(ultimo_usuario['id_usuario']) + 1
    else:
        return 1

def crear_usuario(request):
    
    if request.method == 'POST':
        # Obtén los datos del formulario
        nombre = request.POST['nombre']
        apellido_p = request.POST['apellido_p']
        apellido_m = request.POST['apellido_m']
        matricula = request.POST['matricula']
        carrera = request.POST['carrera']
        grado = request.POST['grado']
        contraseña = request.POST['contraseña']

        
        # Obtiene el último id_usuario y genera el siguiente ID
        db_handle, client = get_db_handle()
        id_usuario = obtener_ultimo_id_usuario(db_handle)
        
        # Crea un nuevo documento de usuario
        usuario = {
            "id_usuario": int(id_usuario),  
            "nombre": nombre,
            "apellido_p": apellido_p,
            "apellido_m": apellido_m,
            "matricula" : matricula,
            "carrera" : carrera,
            "grado" : grado,
            "contraseña": contraseña
        }
        
        # Guarda el documento en la base de datos
        db_handle.usuarios.insert_one(usuario)
        client.close()
        
        return redirect('lista_usuario')
    else:
        return render(request, 'formulario_crear_U.html')
    
def panel_usuario(request):

    
    return render(request, 'panel_usuario.html')



def leer_prestamo(request):
    db_handle, client = get_db_handle()
    prestamos = db_handle.prestamos.find()
    html_template = loader.get_template('detalles_prestamo.html')
    rendered_html = html_template.render({'prestamos': prestamos}, request)
    client.close()  
    return HttpResponse(rendered_html)

def terminar_prestamo(request, id_prestamo):
    db_handle, client = get_db_handle()
    prestamo = db_handle.prestamos.find_one({"id_prestamo": id_prestamo})
    fecha= prestamo["fecha_devolucion"]
    print(fecha)
    if fecha == "":
        fecha_devolucion_iso = date.today().isoformat()
            # Establece el campo 'estado' a 'disponible'
        db_handle.prestamos.update_one(
                {"id_prestamo": id_prestamo},
                {"$set": {"fecha_devolucion": fecha_devolucion_iso}}
            )

        prestamo = db_handle.prestamos.find_one({"id_prestamo": id_prestamo})
        id_libro = prestamo["id_libro"]
        print(id_libro)
        db_handle.libro.update_one(
                {"id_libro": id_libro},
                {"$set": {"estado": "disponible"}}
            )
    client.close()
    return redirect('lista_prestamo')

def prestamos_libros(request):
    db_handle, client = get_db_handle()
    usuarios = db_handle.usuarios.find()
    html_template = loader.get_template('prestamo_libro.html')
    rendered_html = html_template.render({'usuarios': usuarios}, request)
    client.close()
    return HttpResponse(rendered_html)

def reconocer(request):
    datos_componentes = []
    db_handle, client = get_db_handle()
    libros = db_handle.libro.find()
    
    book_images_folder = 'book_images'
    imagen = []
    # Cargar imágenes recortadas de libros y agregar a los datos de entrenamiento
    if os.path.exists(book_images_folder):
        print("Cargando imágenes recortadas de libros...")

        # Cargar o inicializar el modelo
        model_filename = 'trained_model.pkl'
        if os.path.exists(model_filename):
            classifier = joblib.load(model_filename)
            print(f"Modelo cargado desde {model_filename}.")
        else:
            print("Inicializando nuevo modelo...")
            # Ejemplo de datos de entrenamiento inicial (aspect_ratio, area)
            X_train = np.array([[1.0, 100], [0.9, 80], [1.2, 120], [0.8, 70]])
            y_train = np.array([1, 0, 1, 0])  # Etiquetas: 1 para libros, 0 para otros
            train_and_save_model(X_train, y_train, model_filename)
            classifier = joblib.load(model_filename)

        # Iniciar la captura de video desde la cámara
        cap = cv2.VideoCapture(0)  # Usar cámara predeterminada
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Desactivar exposición automática
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 100)
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 100)

        max_images_to_save = 3  # Máximo número de imágenes a guardar
        img_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(gray, 50, 150)
            cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                area = cv2.contourArea(c)
                min_area_threshold = 3000
                if area > min_area_threshold:
                    x, y, w, h = cv2.boundingRect(c)
                    aspect_ratio = float(w) / h
                    prediction = classifier.predict([[aspect_ratio, area]])

                    if prediction == 1:
                        book_img = frame[y:y + h, x:x + w]
                        imagen.append(book_img)
                        cleaned_title = extract_book_title(book_img)
                        print("Título del libro:", cleaned_title)
                        img_counter += 1
                        img_name = os.path.join(book_images_folder, f'{img_counter:07d}.jpg')
                        cv2.imwrite(img_name, book_img)
                        print(f"Imagen del libro guardada como {img_name}")

                        if img_counter >= max_images_to_save:
                            break

            if img_counter >= max_images_to_save:
                sample_images = load_and_preprocess_images(imagen)
                pca = PCA(n_components=2)
                pca.fit(sample_images)
                datos_componentes = pca.transform(sample_images)
                
                cap.release()
                cv2.destroyAllWindows()
                break

            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

    menor_similitud = float('inf')
    libro_encontrado = None

    for libro in libros:
        cadena = libro['caracteristicas'].strip('[]')
        matrices_str = cadena.split('] [')
        matrices_np = [np.fromstring(matriz_str, sep=' ') for matriz_str in matrices_str]

        for i, matriz in enumerate(matrices_np):
            for j, otra_matriz in enumerate(datos_componentes):
                matriz_flatten = matriz.flatten()
                otra_matriz_flatten = otra_matriz.flatten()
                similitud = distance.euclidean(matriz_flatten, otra_matriz_flatten)

                if similitud < menor_similitud:
                    menor_similitud = similitud
                    libro_encontrado = libro

    if menor_similitud < 200 and libro_encontrado:
        print(f"Similitud más baja encontrada: {menor_similitud}")
        print("Libro encontrado")
        usuarios = db_handle.usuarios.find()   
        html_template = loader.get_template('prestamo_libro.html')
        rendered_html = html_template.render({'usuarios': usuarios, 'libro': libro_encontrado}, request)
        client.close()
        return HttpResponse(rendered_html)
    else:
        messages.warning(request, "No se encontró ningún libro con coincidencia.")
        client.close()
        return redirect('manejo_presta') 
     
def manejo_presta(request):
    return render(request, 'Reconocimiento_presta.html')

def obtener_ultimo_id_prestamo(db_handle):
    ultimo_prestamo = db_handle.prestamos.find_one({}, sort=[("id_prestamo", -1)])
    if ultimo_prestamo:
        return int(ultimo_prestamo['id_prestamo']) + 1
    else:
        return 1


def crear_prestamo(request):
    global nombre_user
    if request.method == 'POST':
        db_handle, client = get_db_handle()
        # Obtén los datos del formulario
        fecha_prestamo =date.today().isoformat()
        titulo = request.POST.get('titulo')
        id_libro=db_handle.libro.find_one(titulo)
        print(id_libro)
        fecha_devolucion = ""
        usuario =request.POST['usuario']
        encargado=nombre_user

        
        # Obtiene el último id_usuario y genera el siguiente ID
        
        id_usuario = obtener_ultimo_id_prestamo(db_handle)
        
        # Crea un nuevo documento de usuario
        prestamo = {
            "id_prestamo": int(id_usuario),  
            "fecha_prestamo": fecha_prestamo,
            "id_libro": id_libro['id_libro'],
            "fecha_devolucion": fecha_devolucion,
            "usuario" : usuario,
            "encargado" : encargado
        }
        
        # Guarda el documento en la base de datos
        db_handle.prestamos.insert_one(prestamo)
        result = db_handle.libro.update_one(
            {"id_libro": id_libro['id_libro']},                      # Filtro para encontrar el documento
            {"$set": {"estado": "no disponible"}}  # Operación de actualización
        )

        
        client.close()
        
        return redirect('lista_prestamo')
    else:
        return render(request, 'prestamo_libro.html')

def leer_libro(request):
    db_handle, client = get_db_handle()
    libros = db_handle.libro.find()
    html_template = loader.get_template('detalles_libro.html')
    rendered_html = html_template.render({'libros': libros}, request)
    client.close()  
    return HttpResponse(rendered_html)

def actualizar_libro(request, id_libro):
    if request.method == 'POST':
        # Obtiene los datos actualizados del formulario
        id_libro = int(request.POST['id_libro'])
        nombre = request.POST['nombre']
        apellido_p = request.POST['apellido_p']
        apellido_m = request.POST['apellido_m']
        usuario = request.POST['usuario']
        contraseña = request.POST['contraseña']

        # Actualiza el documento en la colección "libro"
        db_handle, client = get_db_handle()
        db_handle.libro.update_one({"id_libro": id_libro}, {"$set": {
            "nombre": nombre,
            "apellido_p": apellido_p,
            "apellido_m": apellido_m,
            "usuario": usuario,
            "contraseña": contraseña
        }})
        client.close()
        return redirect('lista_libro')
    else:
        # Muestra el formulario de actualización
        db_handle, client = get_db_handle()
        libro = db_handle.libro.find_one({"id_libro": id_libro})
        client.close()
        return render(request, 'formulario_actualizar.html', {'libro': libro})
    
def eliminar_libro( request, id_libro):
    # Elimina el libro de la base de datos utilizando su id_libro
    db_handle, client = get_db_handle()
    db_handle.libro.delete_one({"id_libro": id_libro})

    client.close()
    
    # Redirige al usuario a la lista de encargados después de eliminar exitosamente
    return redirect('lista_libro')

def obtener_ultimo_id_libro(db_handle):
    ultimo_libro = db_handle.libro.find_one({}, sort=[("id_libro", -1)])
    if ultimo_libro:
        return int(ultimo_libro['id_libro']) + 1
    else:
        return 1

def manejo_libro(request):

        return render(request, 'Reconocimiento_libro.html')
    
def reconocer_libro(request):
   # Ruta de la carpeta para las imágenes recortadas de libros
    book_images_folder = 'book_images'
    imagen=[]
    # Cargar imágenes recortadas de libros y agregar a los datos de entrenamiento
    if os.path.exists(book_images_folder):
        print("Cargando imágenes recortadas de libros...")

        # Cargar o inicializar el modelo
        model_filename = 'trained_model.pkl'
        if os.path.exists(model_filename):
            classifier = joblib.load(model_filename)
            print(f"Modelo cargado desde {model_filename}.")
        else:
            print("Inicializando nuevo modelo...")
            # Ejemplo de datos de entrenamiento inicial (aspect_ratio, area)
            X_train = np.array([[1.0, 100], [0.9, 80], [1.2, 120], [0.8, 70]])
            y_train = np.array([1, 0, 1, 0])  # Etiquetas: 1 para libros, 0 para otros
            train_and_save_model(X_train, y_train, model_filename)
            classifier = joblib.load(model_filename)

        # Iniciar la captura de video desde la cámara
        cap = cv2.VideoCapture(0)  # Usar cámara predeterminada

        # Ajustar la exposición automática de la cámara (opcional)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Desactivar exposición automática

        # Ajustar el balance de blancos de la cámara
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 100)
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 100)

        max_images_to_save = 3  # Máximo número de imágenes a guardar
        img_counter = 0

        while True:
            ret, frame = cap.read()  # Capturar fotograma desde la cámara
            if not ret:
                break

            # Convertir el fotograma a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Detectar bordes en la imagen
            edged = cv2.Canny(gray, 50, 150)

            # Encontrar contornos en la imagen
            cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Loop sobre los contornos
            for c in cnts:
                # Calcular el área del contorno
                area = cv2.contourArea(c)

                # Filtrar contornos grandes (libros)
                min_area_threshold = 3000  # Ajusta según sea necesario
                if area > min_area_threshold:
                    # Calcular el rectángulo delimitador
                    x, y, w, h = cv2.boundingRect(c)

                    # Predecir si el contorno representa un libro usando el clasificador
                    aspect_ratio = float(w) / h
                    prediction = classifier.predict([[aspect_ratio, area]])

                    # Si el clasificador predice que es un libro
                    if prediction == 1:
                        # Recortar la región del libro
                        book_img = frame[y:y + h, x:x + w]
                        imagen.append(book_img)

                        # Extraer el título del libro de la región del libro
                        cleaned_title = extract_book_title(book_img)

                        # Mostrar el título del libro extraído
                        print("Título del libro:", cleaned_title)

                        # Guardar la imagen del libro recortada
                        img_counter += 1
                        img_name = os.path.join(book_images_folder, f'{img_counter:07d}.jpg')  # Numeración de 7 dígitos
                        cv2.imwrite(img_name, book_img)
                        print(f"Imagen del libro guardada como {img_name}")

                        if img_counter >= max_images_to_save:
                            break  # Salir del bucle si se alcanza el máximo de imágenes a guardar

            if img_counter >= max_images_to_save:
                sample_images = load_and_preprocess_images(imagen)
                pca = PCA(n_components=2)  # Especificar el número de componentes principales deseados
                pca.fit(sample_images)
                sample_images_pca = pca.transform(sample_images)
                datos = {'sample_images_pca': sample_images_pca}
                cap.release()
                cv2.destroyAllWindows()
                url = reverse('crear_libro', kwargs=datos)
                return redirect(url)
                break  # Salir del bucle si se alcanza el máximo de imágenes a guardar

            # Mostrar el fotograma con los contornos detectados
            cv2.imshow('Frame', frame)
            #time.sleep(0.5)
            #window = gw.getWindowsWithTitle('Frame')[0]
            #window.activate()
            #pyautogui.click(window.left + 10, window.top + 10)
            # Salir del bucle si se presiona 'q'
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                redirect()
                break

        

        return HttpResponse("Proceso completado.")

    else:
        return HttpResponse("No se encontraron imágenes recortadas de libros en la carpeta.")

def crear_libro(request, sample_images_pca):
    cadena_final = sample_images_pca.replace("\n", "")
    cadena_sin_espacios = re.sub(r'\[\s+', '[', cadena_final)
    cadena_sin_espacios = re.sub(r'\s+\]', ']', cadena_sin_espacios)
    print(cadena_sin_espacios)
    if request.method == 'POST':
        titulo = request.POST['titulo']
        autor = request.POST['autor']
        editorial = request.POST['editorial']
        edicion = request.POST['edicion']
        genero = request.POST['genero']
        año = request.POST['ano_publicacion']
        caracteristicas = cadena_sin_espacios
        
        
        db_handle, client = get_db_handle()
        id_usuario = obtener_ultimo_id_libro(db_handle)
        # Actualiza el documento en la colección "libro"
        
        libro = {
            "id_libro": int(id_usuario),  
            "titulo": titulo,
            "autor": autor,
            "editorial": editorial,
            "edicion": edicion,
            "año": año,
            "caracteristicas": caracteristicas,
            "estado" : "disponible",
            "genero": genero
        }
        db_handle.libro.insert_one(libro)
        client.close()
        return redirect('lista_libro')
    else:
        return render(request, 'formulario_crear_L.html')

