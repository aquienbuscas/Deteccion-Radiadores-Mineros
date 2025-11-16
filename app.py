import os
import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Ruta del modelo YOLO (ajústala según tu repo)
MODELO_PATH = "best.pt"

# --- Sidebar (mini menú) ---
st.sidebar.title("Menú")
opcion = st.sidebar.radio("Acción:", ["Evaluar imágenes", "Acerca del proyecto"])

# --- Cargar modelo YOLO ---
@st.cache_resource
def cargar_modelo():
    return YOLO(MODELO_PATH)

model = cargar_modelo()

# --- Función de evaluación ---
def evaluar_imagenes(uploaded_files):
    resultados = []
    for uploaded_file in uploaded_files:
        # Abrir imagen
        image = Image.open(uploaded_file).convert("RGB")
        img_cv = np.array(image)

        # Inferencia
        deteccion = model(img_cv, verbose=False)

        # Extraer datos
        boxes = deteccion[0].boxes
        count = len(boxes)
        confs = [float(b.conf) for b in boxes]
        prom_conf = sum(confs)/count if count > 0 else 0

        resultados.append({
            "imagen": uploaded_file.name,
            "objetos_detectados": count,
            "promedio_confianza": round(prom_conf, 3)
        })

        # Dibujar resultados
        result_img = deteccion[0].plot()
        st.image(result_img, caption=f"Detecciones en {uploaded_file.name}", use_column_width=True)

    return pd.DataFrame(resultados)

# --- Lógica principal ---
if opcion == "Evaluar imágenes":
    st.title("Evaluación de imágenes con YOLO")
    uploaded_files = st.file_uploader("Sube una o varias imágenes", type=["jpg","jpeg","png"], accept_multiple_files=True)

    if uploaded_files:
        df = evaluar_imagenes(uploaded_files)
        st.subheader("Resumen general")
        st.dataframe(df)

elif opcion == "Acerca del proyecto":
    st.title("Información del proyecto")
    st.write("""
    Este proyecto aplica un modelo YOLO entrenado con Roboflow para detección de radiadores.
    - Se pueden subir imágenes desde el menú.
    - El modelo evalúa y muestra las clases detectadas.
    """)
