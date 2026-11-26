"""Detección de objetos en tiempo real con YOLO11 + Streamlit.

Ejecutar en local:   streamlit run app.py
"""
import threading
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO

from detector import TRADUCCION, Estabilizador, detecciones_simples, dibujar

st.set_page_config(page_title="Detección de objetos", page_icon="🎥", layout="wide")


# ---------------------------------------------------------------- Modelo
@st.cache_resource(show_spinner="Descargando/cargando modelo...")
def cargar_modelo(nombre):
    return YOLO(nombre)  # descarga los pesos la primera vez


def ice_servers():
    """Servidores STUN/TURN. En la nube hace falta TURN (ver README / secrets)."""
    try:
        if "ice_servers" in st.secrets:
            return [dict(s) for s in st.secrets["ice_servers"]]
    except Exception:
        pass
    return [{"urls": ["stun:stun.l.google.com:19302"]}]


# ---------------------------------------------------------------- Barra lateral
st.sidebar.title("⚙️ Ajustes")
nombre_modelo = st.sidebar.selectbox(
    "Modelo", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"], index=0,
    help="n = más rápido, m = más preciso. En Streamlit Cloud (solo CPU) usa n o s.")
conf_mostrar = st.sidebar.slider("Confianza mínima para mostrar", 0.20, 0.90, 0.50, 0.05)
min_frames = st.sidebar.slider("Fotogramas para confirmar un objeto", 1, 10, 3)
suavizado = st.sidebar.slider("Suavizado de cajas", 0.0, 0.9, 0.6, 0.1)
espejo = st.sidebar.checkbox("Modo espejo (cámara frontal)", value=True)

modelo = cargar_modelo(nombre_modelo)
nombres = modelo.names
opciones = {TRADUCCION.get(n, n): i for i, n in nombres.items()}
elegidas = st.sidebar.multiselect("Detectar solo estas clases (vacío = las 80)", sorted(opciones))
clases = [opciones[c] for c in elegidas] or None


# ---------------------------------------------------------------- Procesador de vídeo
class ProcesadorYOLO(VideoProcessorBase):
    def __init__(self, nombre_modelo):
        # Cada usuario/sesión tiene su propio modelo y tracker (no se mezclan).
        self.modelo = YOLO(nombre_modelo)
        self.estab = Estabilizador()
        self.lock = threading.Lock()
        self.espejo = True
        self.clases = None
        self.fps = 0.0
        self._t = time.time()

    def configurar(self, conf_mostrar, min_frames, suavizado, espejo, clases):
        with self.lock:
            self.estab.conf_mostrar = conf_mostrar
            self.estab.min_frames = min_frames
            self.estab.suavizado = suavizado
            self.espejo = espejo
            self.clases = clases

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            if self.espejo:
                img = cv2.flip(img, 1)
            res = self.modelo.track(img, persist=True, tracker="bytetrack.yaml", conf=0.30,
                                    iou=0.5, imgsz=640, classes=self.clases, verbose=False)[0]
            dets = self.estab.actualizar(res, self.modelo.names)
        out = dibujar(img, dets)

        ahora = time.time()
        inst = 1.0 / max(ahora - self._t, 1e-3)
        self.fps = inst if not self.fps else 0.9 * self.fps + 0.1 * inst
        self._t = ahora
        cv2.putText(out, f"{self.fps:.1f} FPS", (10, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(out, format="bgr24")


# ---------------------------------------------------------------- Interfaz
st.title("🎥 Detección de objetos en tiempo real")
st.caption("YOLO11 + seguimiento ByteTrack + filtro de estabilidad · 80 clases en español")

modo = st.radio("Modo", ["📹 Cámara en vivo", "📸 Tomar foto", "🖼️ Subir imagen"], horizontal=True)

if modo == "📹 Cámara en vivo":
    ctx = webrtc_streamer(
        key=f"yolo-{nombre_modelo}",          # al cambiar de modelo se reinicia el procesador
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": ice_servers()},
        video_processor_factory=lambda: ProcesadorYOLO(nombre_modelo),
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )
    if ctx.video_processor:
        ctx.video_processor.configurar(conf_mostrar, min_frames, suavizado, espejo, clases)
    st.info("Pulsa **START** y acepta el permiso de la cámara. Si se queda cargando en la nube, "
            "configura un servidor TURN (ver README).")

else:
    if modo == "📸 Tomar foto":
        archivo = st.camera_input("Toma una foto")
    else:
        archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "webp"])

    if archivo is not None:
        img = cv2.imdecode(np.frombuffer(archivo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error("No se pudo leer la imagen.")
        else:
            res = modelo.predict(img, conf=0.25, iou=0.5, imgsz=640, classes=clases, verbose=False)[0]
            dets = detecciones_simples(res, nombres, conf_mostrar)
            st.image(dibujar(img, dets)[:, :, ::-1])
            if dets:
                st.write("**Objetos detectados:** " + ", ".join(d["t"] for d in dets))
            else:
                st.warning("No se detectaron objetos con esa confianza.")
