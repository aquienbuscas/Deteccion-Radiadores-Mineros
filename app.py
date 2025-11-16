import streamlit as st
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

# ==============================
# ⚙️ CONFIGURACIÓN INICIAL
# ==============================
MODELO_PATH = "best.pt"

# Logo en la esquina superior izquierda
st.sidebar.image("logo.png", use_column_width=True)

# Menú lateral
st.sidebar.title("Menú")
opcion = st.sidebar.radio("Acción:", ["Evaluar imágenes", "Resultados previos", "Acerca del proyecto"])

# ==============================
# 🧠 CARGA DEL MODELO
# ==============================
@st.cache_resource
def cargar_modelo():
    return YOLO(MODELO_PATH)

model = cargar_modelo()

# ==============================
# 🔍 FUNCIÓN DE EVALUACIÓN
# ==============================
def evaluar_imagenes(uploaded_files):
    resultados = []
    imagenes_procesadas = []  # lista para guardar las imágenes con detecciones

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

        # Guardar imagen procesada
        result_img = deteccion[0].plot()
        imagenes_procesadas.append((uploaded_file.name, result_img))

        # Mostrar en pantalla inmediatamente
        st.image(result_img, caption=f"Detecciones en {uploaded_file.name}", use_column_width=True)

    return pd.DataFrame(resultados), imagenes_procesadas


# ==============================
# ▶️ LÓGICA PRINCIPAL
# ==============================
if opcion == "Evaluar imágenes":
    st.title("Evaluación de imágenes con YOLO")
    uploaded_files = st.file_uploader("Sube una o varias imágenes", type=["jpg","jpeg","png"], accept_multiple_files=True)

    if uploaded_files:
        df, imgs = evaluar_imagenes(uploaded_files)
        st.subheader("Resumen general")
        st.dataframe(df)

        # Guardar resultados en session_state
        st.session_state["resultados"] = df
        st.session_state["imagenes"] = imgs


elif opcion == "Resultados previos":
    st.title("Resultados guardados")
    if "resultados" in st.session_state and st.session_state["resultados"] is not None:
        st.dataframe(st.session_state["resultados"])

        if "imagenes" in st.session_state and st.session_state["imagenes"]:
            st.subheader("Imágenes analizadas previamente")
            for nombre, img in st.session_state["imagenes"]:
                st.image(img, caption=f"Detecciones en {nombre}", use_column_width=True)
    else:
        st.info("No hay resultados previos. Evalúa imágenes primero.")

elif opcion == "Acerca del proyecto":
    st.title("Información del proyecto")
    st.write("""
    Este proyecto aplica un modelo YOLO entrenado con Roboflow para detección de objetos.
    - Se pueden subir imágenes desde el menú.
    - El modelo evalúa y muestra las clases detectadas.
    - Los resultados se guardan y pueden consultarse en la sección 'Resultados previos'.
    """)
