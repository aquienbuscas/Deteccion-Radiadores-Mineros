import streamlit as st
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import torch

# ==============================
# ⚙️ CONFIGURACIÓN INICIAL
# ==============================
MODELO_PATH = "best.pt"

# Logo en la esquina superior izquierda
st.sidebar.image("logo.png", use_column_width=True)

# Menú lateral
st.sidebar.title("Menú")
opcion = st.sidebar.radio("Acción:", ["Evaluar imágenes", "Resultados previos", "Acerca del proyecto"])

# Slider para ajustar confianza
conf_val = st.sidebar.slider("Umbral de confianza", 0.1, 1.0, 0.4, 0.05)  # default 0.4

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
def evaluar_imagenes(uploaded_files, conf_threshold=0.4, iou_threshold=0.5):
    resultados = []
    imagenes_procesadas = []

    for uploaded_file in uploaded_files:
        # Abrir imagen
        image = Image.open(uploaded_file).convert("RGB")
        img_cv = np.array(image)

        # Inferencia con umbral y NMS
        deteccion = model(img_cv, conf=conf_threshold, iou=iou_threshold, verbose=False)

        # Filtrar cajas usando NMS manual por si acaso
        boxes_array = np.array([box.xyxy[0].cpu().numpy() for box in deteccion[0].boxes])
        scores = np.array([float(box.conf) for box in deteccion[0].boxes])
        if len(boxes_array) > 0:
            keep = torch.ops.torchvision.nms(torch.tensor(boxes_array, dtype=torch.float32),
                                             torch.tensor(scores),
                                             iou=iou_threshold)
            boxes_filtradas = [deteccion[0].boxes[int(k)] for k in keep]
        else:
            boxes_filtradas = []

        # Datos resumen
        resultados.append({
            "imagen": uploaded_file.name,
            "objetos_detectados": len(boxes_filtradas)
        })

        # -----------------------------
        # Dibujar cajas filtradas
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

        # Mostrar inmediatamente
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
