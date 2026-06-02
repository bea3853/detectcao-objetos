import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os

# Desativa logs internos e threads de telemetria do Ultralytics no terminal
os.environ["YOLO_VERBOSE"] = "False"

# 1. Configuração da página do Streamlit
st.set_page_config(page_title="Scanner com Yolo", layout="centered")

# 2. Carregamento do modelo YOLOv8 com cache do Streamlit
@st.cache_resource
def carregar_modelo():
    """
    Carrega o modelo YOLOv8 na memória. O decorador cache_resource garante
    que o modelo seja carregado apenas uma vez, otimizando o deploy no Render.
    """
    return YOLO("yolov8n.pt")

model = carregar_modelo()

st.title("Scanner com YOLOv8")
st.write("Protótipo leve e otimizado para detecção de objetos em tempo real.")

# Inicialização do estado da câmera no session_state
if "rodando" not in st.session_state:
    st.session_state.rodando = False

# 4. Interface com botões para controlar o fluxo da câmera
col1, col2 = st.columns(2)
with col1:
    if st.button("Ligar Câmera", type="primary"):
        st.session_state.rodando = True
with col2:
    if st.button("Desligar Câmera"):
        st.session_state.rodando = False

# 3. Bloco lógico de acesso à câmera e detecção de objetos
if st.session_state.rodando:
    # Captura o frame de forma segura através do componente nativo do Streamlit
    imagem_camera = st.camera_input("Posicione o objeto em frente à câmera")

    if imagem_camera is not None:
        # Conversão do buffer de imagem enviado pelo navegador para matriz OpenCV (BGR)
        bytes_data = imagem_camera.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Predição síncrona na thread principal para evitar perda de ScriptRunContext
        resultados = model.predict(source=cv2_img, verbose=False, stream=False)

        # Renderização das caixas delimitadoras sobre a imagem original
        imagem_anotada = resultados[0].plot()

        # Conversão de BGR para RGB (padrão exigido pelo Streamlit)
        imagem_anotada_rgb = cv2.cvtColor(imagem_anotada, cv2.COLOR_BGR2RGB)

        # Exibição do resultado final na interface web
        st.image(imagem_anotada_rgb, caption="Objetos Detectados", use_container_width=True)
else:
    st.info("Clique em 'Ligar Câmera' para iniciar o scanner.")