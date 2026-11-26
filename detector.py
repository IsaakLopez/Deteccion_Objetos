"""Lógica de detección compartida: traducción, colores, estabilizador y dibujo.

Es la misma lógica del notebook de Colab, separada de la parte de cámara
(que en Streamlit se hace con streamlit-webrtc en lugar de google.colab).
"""
import colorsys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TRADUCCION = {
    'person': 'Persona', 'bicycle': 'Bicicleta', 'car': 'Auto', 'motorcycle': 'Moto',
    'airplane': 'Avión', 'bus': 'Autobús', 'train': 'Tren', 'truck': 'Camión', 'boat': 'Barco',
    'traffic light': 'Semáforo', 'fire hydrant': 'Hidrante', 'stop sign': 'Señal de alto',
    'parking meter': 'Parquímetro', 'bench': 'Banca', 'bird': 'Pájaro', 'cat': 'Gato',
    'dog': 'Perro', 'horse': 'Caballo', 'sheep': 'Oveja', 'cow': 'Vaca', 'elephant': 'Elefante',
    'bear': 'Oso', 'zebra': 'Cebra', 'giraffe': 'Jirafa', 'backpack': 'Mochila',
    'umbrella': 'Paraguas', 'handbag': 'Bolso', 'tie': 'Corbata', 'suitcase': 'Maleta',
    'frisbee': 'Frisbee', 'skis': 'Esquís', 'snowboard': 'Snowboard', 'sports ball': 'Pelota',
    'kite': 'Cometa', 'baseball bat': 'Bate', 'baseball glove': 'Guante de béisbol',
    'skateboard': 'Patineta', 'surfboard': 'Tabla de surf', 'tennis racket': 'Raqueta',
    'bottle': 'Botella', 'wine glass': 'Copa', 'cup': 'Taza', 'fork': 'Tenedor',
    'knife': 'Cuchillo', 'spoon': 'Cuchara', 'bowl': 'Tazón', 'banana': 'Banano',
    'apple': 'Manzana', 'sandwich': 'Sándwich', 'orange': 'Naranja', 'broccoli': 'Brócoli',
    'carrot': 'Zanahoria', 'hot dog': 'Hot dog', 'pizza': 'Pizza', 'donut': 'Dona',
    'cake': 'Pastel', 'chair': 'Silla', 'couch': 'Sofá', 'potted plant': 'Planta',
    'bed': 'Cama', 'dining table': 'Mesa', 'toilet': 'Inodoro', 'tv': 'Televisor',
    'laptop': 'Laptop', 'mouse': 'Mouse', 'remote': 'Control remoto', 'keyboard': 'Teclado',
    'cell phone': 'Celular', 'microwave': 'Microondas', 'oven': 'Horno', 'toaster': 'Tostadora',
    'sink': 'Lavabo', 'refrigerator': 'Refrigerador', 'book': 'Libro', 'clock': 'Reloj',
    'vase': 'Florero', 'scissors': 'Tijeras', 'teddy bear': 'Peluche',
    'hair drier': 'Secadora de pelo', 'toothbrush': 'Cepillo de dientes',
}

def _cargar_fuente(tam=17):
    """Fuente TrueType con tildes y ñ (DejaVu viene de packages.txt en Streamlit Cloud)."""
    for ruta in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(ruta, tam)
        except OSError:
            pass
    try:  # respaldo: fuente que descarga ultralytics
        from ultralytics.utils.checks import check_font
        return ImageFont.truetype(str(check_font("Arial.Unicode.ttf")), tam)
    except Exception:
        return ImageFont.load_default()


FUENTE = _cargar_fuente()


def nombre_es(nombres_modelo, cls):
    en = nombres_modelo[int(cls)]
    return TRADUCCION.get(en, en)


def color_clase(cls):
    """Color fijo y bien distinto para cada clase (RGB)."""
    r, g, b = colorsys.hsv_to_rgb((int(cls) * 0.618034) % 1.0, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def dibujar(img_bgr, dets):
    """Dibuja cajas y etiquetas. dets = [{'b': (x1,y1,x2,y2), 't': texto, 'c': (r,g,b)}]."""
    img = Image.fromarray(img_bgr[:, :, ::-1])
    d = ImageDraw.Draw(img)
    for det in dets:
        x1, y1, x2, y2 = det['b']
        d.rectangle((x1, y1, x2, y2), outline=det['c'], width=3)
        l, t, r, b = d.textbbox((0, 0), det['t'], font=FUENTE)
        w, h = r - l + 10, b - t + 8
        ty = max(0, y1 - h)
        tx = max(0, min(x1, img.width - w))   # que la etiqueta no se salga de la imagen
        d.rectangle((tx, ty, tx + w, ty + h), fill=det['c'])
        d.text((tx + 5, ty + 4 - t), det['t'], fill=(0, 0, 0), font=FUENTE)
    return np.ascontiguousarray(np.array(img)[:, :, ::-1])


def detecciones_simples(res, nombres_modelo, conf_min=0.5):
    """Para fotos sueltas (sin seguimiento)."""
    dets = []
    b = res.boxes
    for xyxy, cls, cf in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()):
        if cf < conf_min:
            continue
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        dets.append({'b': (x1, y1, x2, y2), 't': f'{nombre_es(nombres_modelo, cls)} {cf * 100:.0f}%',
                     'c': color_clase(cls)})
    return dets


class Estabilizador:
    """Confirma objetos en varios fotogramas, vota la clase y suaviza la caja."""

    def __init__(self, conf_mostrar=0.5, min_frames=3, max_perdido=6, suavizado=0.6):
        self.conf_mostrar = conf_mostrar
        self.min_frames = min_frames
        self.max_perdido = max_perdido
        self.suavizado = suavizado
        self.t = {}
        self.n = 0

    def actualizar(self, res, nombres_modelo):
        self.n += 1
        b = res.boxes
        if b.id is not None:
            for xyxy, cls, cf, tid in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int),
                                          b.conf.cpu().numpy(), b.id.cpu().numpy().astype(int)):
                s = self.t.get(tid)
                if s is None:
                    s = self.t[tid] = {'box': xyxy, 'conf': float(cf), 'hits': 0,
                                       'votos': defaultdict(float)}
                else:
                    a = self.suavizado
                    s['box'] = a * s['box'] + (1 - a) * xyxy
                    s['conf'] = 0.7 * s['conf'] + 0.3 * float(cf)
                s['hits'] += 1
                s['visto'] = self.n
                s['votos'][cls] += float(cf)

        dets = []
        for tid in list(self.t):
            s = self.t[tid]
            if self.n - s['visto'] > self.max_perdido:
                del self.t[tid]
                continue
            if s['hits'] < self.min_frames or s['conf'] < self.conf_mostrar:
                continue
            cls = max(s['votos'], key=s['votos'].get)
            x1, y1, x2, y2 = (int(v) for v in s['box'])
            dets.append({'b': (x1, y1, x2, y2),
                         't': f'{nombre_es(nombres_modelo, cls)} {s["conf"] * 100:.0f}%',
                         'c': color_clase(cls)})
        return dets
