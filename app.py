import streamlit as st
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import os
import gdown
from ultralytics import YOLO
import requests


# ==============================
# ⚙️ CONFIGURACIÓN DEL MODELO
# ==============================
MODEL_PATH = "best.pt"
HF_URL = "https://huggingface.co/Remberto/detector-radiadores/tree/main/best.pt"  # reemplaza con tu enlace de Hugging Face

# Descargar modelo si no existe localmente
if not os.path.exists(MODEL_PATH):
    try:
        st.info("Descargando modelo desde Hugging Face… esto puede tardar unos segundos")
        response = requests.get(HF_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(MODEL_PATH, 'wb') as f:
            for data in response.iter_content(1024):
                f.write(data)
        st.success("Modelo descargado correctamente")
    except Exception as e:
        st.error(f"No se pudo descargar el modelo: {e}")
        st.stop()

# ==============================
# Cargar modelo YOLO
# ==============================
@st.cache_resource
def cargar_modelo():
    return YOLO(MODEL_PATH)

model = cargar_modelo()


# ==============================
# 🔍 FUNCIÓN DE EVALUACIÓN
# ==============================
def evaluar_imagenes(uploaded_files, conf_threshold=0.4, iou_threshold=0.5):
    resultados = []
    imagenes_procesadas = []

    for uploaded_file in uploaded_files:
        # Abrir imagen
        image = Image.open(uploaded_file).convert("RGB")
        img_cv = np.array(image)

        # Inferencia con NMS interno de YOLO
        deteccion = model(img_cv, conf=conf_threshold, iou=iou_threshold, verbose=False)
        boxes_filtradas = deteccion[0].boxes  # cajas filtradas automáticamente por NMS

        # Datos resumen
        resultados.append({
            "imagen": uploaded_file.name,
            "objetos_detectados": len(boxes_filtradas)
        })

        # -----------------------------
        # Dibujar cajas filtradas con nombres de clase
        # -----------------------------
        img_disp = img_cv.copy()
        for box, cls in zip(boxes_filtradas, [b.cls for b in boxes_filtradas]):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            label = model.names[int(cls)]

            # Rectángulo
            color = (0, 255, 0)
            cv2.rectangle(img_disp, (x1, y1), (x2, y2), color, 2)

            # Texto con fondo
            font_scale = 0.5
            thickness = 1
            ((w, h), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(img_disp, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
            cv2.putText(img_disp, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        img_disp = Image.fromarray(img_disp)
        imagenes_procesadas.append((uploaded_file.name, img_disp))

        # Mostrar en pantalla
        st.image(img_disp, caption=f"Detecciones en {uploaded_file.name}", use_column_width=True)

    return pd.DataFrame(resultados), imagenes_procesadas

# ==============================
# ▶️ LÓGICA PRINCIPAL
# ==============================
if opcion == "Evaluar imágenes":
    st.title("Evaluación de imágenes con YOLO")
    uploaded_files = st.file_uploader("Sube una o varias imágenes", type=["jpg","jpeg","png"], accept_multiple_files=True)

    if uploaded_files:
        df, imgs = evaluar_imagenes(uploaded_files, conf_threshold=conf_val, iou_threshold=0.5)
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
