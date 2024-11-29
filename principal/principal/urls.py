"""
URL configuration for Intento project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from modular2.views import login , leer_encargado, actualizar_encargado, eliminar_encargado, crear_encargado,leer_admin, actualizar_admin, eliminar_admin, crear_admin,panel_admin,panel_encargado,leer_usuario,panel_usuario,crear_usuario,eliminar_usuario,actualizar_usuario,prestamos_libros,leer_prestamo,terminar_prestamo,leer_libro,crear_libro,eliminar_libro,actualizar_libro,reconocer_libro,manejo_libro,reconocer,crear_prestamo,manejo_presta

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', login, name='Login'),

    path('panel_control_encargado/', panel_encargado, name='panel_control_encargado'),
    path('lista_encargado/', leer_encargado, name='lista_encargado'),
    path('actualizar_encargado/<int:id_encargado>/', actualizar_encargado, name='actualizar_encargado'),
    path('eliminar_encargado/<int:id_encargado>/', eliminar_encargado, name='eliminar_encargado'),
    path('crear_encargado/', crear_encargado, name='crear_encargado'),

    path('panel_control_admin/', panel_admin, name='panel_control_admin'),
    path('lista_admin/', leer_admin, name='lista_admin'),
    path('actualizar_admin/<int:id_admin>/', actualizar_admin, name='actualizar_admin'),
    path('eliminar_admin/<int:id_admin>/', eliminar_admin, name='eliminar_admin'),
    path('crear_admin/', crear_admin, name='crear_admin'),

    path('panel_control_usuario/', panel_usuario, name='panel_control_usuario'),
    path('lista_usuario/', leer_usuario, name='lista_usuario'),
    path('actualizar_usuario/<int:id_usuario>/', actualizar_usuario, name='actualizar_usuario'),
    path('eliminar_usuario/<int:id_usuario>/', eliminar_usuario, name='eliminar_usuario'),
    path('crear_usuario/', crear_usuario, name='crear_usuario'),


    path('prestamos_libros/', prestamos_libros, name='prestamos'),
    path('lista_prestamo/', leer_prestamo, name='lista_prestamo'),
    path('crear_prestamo/', crear_prestamo, name='crear_prestamo'),
    path('reconocer/', reconocer, name='reconocer'),
    path('terminar_prestamo/<int:id_prestamo>/', terminar_prestamo, name='terminar_prestamo'),
    path('manejo_presta/', manejo_presta, name='manejo_presta'),

    path('lista_libro/', leer_libro, name='lista_libro'),
    path('actualizar_libro/<int:id_libro>/', actualizar_libro, name='actualizar_libro'),
    path('eliminar_libro/<int:id_libro>/', eliminar_libro, name='eliminar_libro'),
    path('crear_libro/<str:sample_images_pca>/', crear_libro, name='crear_libro'),
    path('detect_books/', reconocer_libro, name='detect_books'),
    path('manejo_libro/', manejo_libro, name='manejo_libro'),
]
