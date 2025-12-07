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
        # Inferencia con nuevo umbral
        deteccion = model(img_cv, conf=0.4, iou=0.5, verbose=False)


        # Extraer datos
        boxes = deteccion[0].boxes
        count = len(boxes)

        # Para mostrar información resumida por imagen
        resultados.append({
            "imagen": uploaded_file.name,
            "objetos_detectados": count
        })

        # -----------------------------
        # Dibujar manualmente las cajas con nombres de clase
        # -----------------------------
        img_disp = img_cv.copy()
        for box, cls in zip(boxes.xyxy, boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]  # nombre de la clase

            # Rectángulo
            color = (0, 255, 0)
            cv2.rectangle(img_disp, (x1, y1), (x2, y2), color, 2)

            # Texto
            font_scale = 0.5
            thickness = 1
            ((w, h), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(img_disp, (x1, y1 - h - 4), (x1 + w, y1), color, -1)  # fondo del texto
            cv2.putText(img_disp, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Convertir de nuevo a PIL para Streamlit
        img_disp = Image.fromarray(img_disp)
        imagenes_procesadas.append((uploaded_file.name, img_disp))

        # Mostrar en pantalla inmediatamente
        st.image(img_disp, caption=f"Detecciones en {uploaded_file.name}", use_column_width=True)

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
