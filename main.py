import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from datetime import datetime
from angulos import (
    angulo_tronco_vertical,
    angulo_cuello,
    angulo_brazo_tronco,
    angulo_antebrazo,
    angulo_muneca,
    calcular_angulo_3puntos,
)
from reba import calcular_reba_completo
from repeticiones import DetectorRepeticiones, FiltroAngulos, GestorOclusiones


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

ALTURA_REFERENCIA_CM = 170  # Altura de referencia estándar en cm
CAMARA_ID = 0               # ID de la cámara (0 = webcam por defecto)
GUARDAR_CSV = True          # Guardar datos en CSV automáticamente
MOSTRAR_ESQUELETO = True     # Mostrar landmarks de MediaPipe
VENTANA_AUTOCORRELACION_S = 10.0 # Ventana de tiempo para analizar repeticiones (en segundos)
ALPHA_EMA = 0.3 # Coeficiente de suavizado para filtro EMA de ángulos (0-1)
UMBRAL_VISIBILIDAD = 0.5 # Umbral de visibilidad para considerar un landmark como válido (0-1)


# ==============================================================================
# CALIBRADOR
# ==============================================================================

class Calibrador:
    """
    Calibra la escala de la imagen basándose en la altura conocida del sujeto.
    
    La calibración mide la distancia en píxeles entre la cabeza (nariz/oreja)
    y los pies (tobillo), y la compara con la altura real introducida.
    
    Esto permite:
    1. Normalizar medidas independientemente de la distancia a la cámara.
    2. Ajustar umbrales REBA si la altura afecta la escala de movimientos.
    """

    def __init__(self):
        self.calibrado = False
        self.px_por_cm = None
        self.altura_sujeto_cm = ALTURA_REFERENCIA_CM
        self.altura_px = None
        self.factor_escala = 1.0 # Ratio altura_sujeto / altura_referencia
        self.muestras_altura_px = []
        self.n_muestras = 30 # Promedia 30 frames para más estabilidad
        self.calibrando = False
        self.inicio_calibracion = None

    def iniciar_calibracion(self, altura_cm):
        """Inicia el proceso de calibración con la altura del sujeto en cm."""
        self.altura_sujeto_cm = altura_cm
        self.muestras_altura_px = []
        self.calibrando = True
        self.calibrado = False
        self.inicio_calibracion = time.time()
        print(f"[Calibración] Iniciando para sujeto de {altura_cm} cm...")

    def actualizar(self, landmarks, h_imagen):
        """
        Añade una muestra de altura en píxeles.
        Usa nariz (landmark 0) como punto superior y los tobillos como inferior.
        """
        if not self.calibrando:
            return
        try:
            nariz = landmarks[0]
            tobillo_izq = landmarks[27]
            tobillo_der = landmarks[28]

            # Punto más alto: nariz
            y_superior = nariz.y * h_imagen
            # Punto más bajo: promedio de tobillos
            y_inferior = ((tobillo_izq.y + tobillo_der.y) / 2) * h_imagen
            altura_px_muestra = abs(y_inferior - y_superior)
            
            # Solo añadir si la persona está completamente visible
            if (nariz.visibility > 0.7 and
                    tobillo_izq.visibility > 0.5 and
                    tobillo_der.visibility > 0.5 and
                    altura_px_muestra > 100):
                self.muestras_altura_px.append(altura_px_muestra)

            if len(self.muestras_altura_px) >= self.n_muestras:
                self._finalizar_calibracion()

        except Exception as e:
            print(f"[Calibración] Error: {e}")

    def _finalizar_calibracion(self):
        """Calcula los parámetros de calibración finales."""
        self.altura_px = np.median(self.muestras_altura_px)
        self.px_por_cm = self.altura_px / self.altura_sujeto_cm
        self.factor_escala = self.altura_sujeto_cm / ALTURA_REFERENCIA_CM
        self.calibrado = True
        self.calibrando = False
        print(f"[Calibración] Completada:")
        print(f"  Altura en px: {self.altura_px:.1f}")
        print(f"  px/cm: {self.px_por_cm:.2f}")
        

    def estado(self):
        """Devuelve el estado actual de la calibración como string."""
        if self.calibrado:
            return f"CAL OK ({self.altura_sujeto_cm}cm)"
        elif self.calibrando:
            return f"Calibrando... {len(self.muestras_altura_px)}/{self.n_muestras}"
        else:
            return "Sin calibrar [ESPACIO]"


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def seleccionar_lado(landmarks):
    """
    Selecciona el lado (izquierdo/derecho) más visible para el análisis.
    Compara la visibilidad media de hombro, codo, muñeca, cadera, rodilla.
    """
    indices_izq = [11, 13, 15, 23, 25, 27] # hombro, codo, muñeca, cadera, rodilla, tobillo
    indices_der = [12, 14, 16, 24, 26, 28]
    vis_izq = np.mean([landmarks[i].visibility for i in indices_izq])
    vis_der = np.mean([landmarks[i].visibility for i in indices_der])
    return "izq" if vis_izq >= vis_der else "der"


def extraer_landmarks(landmarks, lado):
    """
    Extrae los landmarks relevantes según el lado seleccionado.
    Devuelve un diccionario con coordenadas normalizadas [x, y].
    """
    if lado == "izq":
        idx = {"hombro": 11, "codo": 13, "muneca": 15,
               "indice": 19, "cadera": 23, "rodilla": 25,
               "tobillo": 27, "oreja": 7} # Índice de la mano
    else:
        idx = {"hombro": 12, "codo": 14, "muneca": 16,
               "indice": 20, "cadera": 24, "rodilla": 26,
               "tobillo": 28, "oreja": 8}
    idx["nariz"] = 0 # Nariz es siempre 0

    puntos = {}
    for nombre, i in idx.items():
        lm = landmarks[i]
        puntos[nombre] = [lm.x, lm.y]
        puntos[nombre + "_vis"] = lm.visibility
    return puntos


def punto_px(punto_norm, w, h):
    """Convierte coordenadas normalizadas a píxeles."""
    return (int(punto_norm[0] * w), int(punto_norm[1] * h))


def calcular_angulo_rodilla_reba(puntos):
    """
    Calcula la flexión de la rodilla para REBA.
    0° = pierna recta, valores positivos = flexión.
    """
    ang_3p = calcular_angulo_3puntos(
        puntos["cadera"], puntos["rodilla"], puntos["tobillo"])
    # El ángulo 3 puntos da 180° cuando la pierna está recta, y disminuye al flexionar.
    return abs(180 - ang_3p)


# ==============================================================================
# VISUALIZACIÓN
# ==============================================================================

def dibujar_panel_info(frame, reba_result, angulos_dict, lado, calibrador,
                       rep_resultado, w, h):
    """
    Dibuja el panel de información ergonómica en el frame.
    """
    # Panel de fondo semitransparente en la parte izquierda
    overlay = frame.copy()
    panel_w = 340
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    C_TIT  = (200, 200, 255) #color título
    C_OK   = (100, 255, 100) #color ok
    C_WARN = (50, 200, 255) #color warning
    C_HIGH = (50, 100, 255) #color alto
    C_TXT  = (220, 220, 220) #color texto
    C_REP  = (50, 255, 200) #color repetición

    def t(msg, x, y, color=C_TXT, s=0.45, g=1):
        cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    s, color, g, cv2.LINE_AA)

    def linea(y):
        cv2.line(frame, (10, y), (panel_w - 10, y), (80, 80, 80), 1)

    y = 22
    # Título
    t("ANÁLISIS", 10, y, C_TIT, 0.6, 2)
    y += 6; linea(y); y += 14

    # Estado calibración
    t(f"Calibracion: {calibrador.estado()}", 10, y,
      C_OK if calibrador.calibrado else (50, 150, 255))
    y += 13
    t(f"Lado: {lado.upper()}", 10, y)
    y += 16; linea(y); y += 12

    # Ángulos
    t("ÁNGULOS (°)", 10, y, C_TIT, 0.47, 1)
    y += 13
    for nombre, clave in [("Tronco/vertical", "tronco"),
                           ("Cuello/tronco", "cuello"),
                           ("Brazo/tronco", "brazo"),
                           ("Antebrazo (codo)", "antebrazo"),
                           ("Muñeca", "muneca"),
                           ("Rodilla (flex.)", "rodilla")]:
        val = angulos_dict.get(clave, 0)
        t(f"  {nombre}: {val:.1f}", 10, y, C_TXT, 0.4)
        y += 12
    y += 4; linea(y); y += 12

    # Scores individuales
    t("SCORES INDIVIDUALES", 10, y, C_TIT, 0.47, 1)
    y += 13
    items = [("Tronco", "s_tronco", 4), ("Cuello", "s_cuello", 3),
             ("Piernas", "s_piernas", 4), ("Brazo", "s_brazo", 6),
             ("Antebrazo", "s_antebrazo", 2), ("Muñeca", "s_muneca", 3)]
    
    for nombre, key, max_s in items:
        # Color según nivel del score
        score = reba_result[key]
        ratio = score / max_s
        sc = C_OK if ratio <= 0.4 else (C_WARN if ratio <= 0.7 else C_HIGH)
        # Barra de progreso
        bx, bw_bar = 180, 110
        cv2.rectangle(frame, (bx, y - 8), (bx + bw_bar, y), (60, 60, 60), -1)
        cv2.rectangle(frame, (bx, y - 8), (bx + int(bw_bar * ratio), y), sc, -1)
        t(f"  {nombre}: {score}/{max_s}", 10, y, sc, 0.4)
        y += 13
    y += 4; linea(y); y += 12

    # Sección repeticiones
    t("MOVIMIENTO REPETITIVO", 10, y, C_REP, 0.47, 1)
    y += 13
    es_rep = rep_resultado["es_repetitivo"]
    rpm = rep_resultado["frecuencia_rpm"]
    confianza = rep_resultado["confianza"]
    mod_rep = rep_resultado["modificador_reba"]

    col_rep = C_HIGH if es_rep else C_OK
    t(f"  Estado: {'SI — DETECTADO' if es_rep else 'No detectado'}", 10, y, col_rep, 0.42)
    y += 12
    if es_rep:
        t(f"  Frecuencia: {rpm:.1f} rep/min", 10, y, col_rep, 0.4)
        y += 12
    t(f"  Confianza ACF: {confianza:.2f}", 10, y, C_TXT, 0.4)
    y += 12
    t(f"  Mod. actividad REBA: +{mod_rep}", 10, y,
      C_HIGH if mod_rep > 0 else C_TXT, 0.4)
    y += 14; linea(y); y += 12

    # Score final
    t(f"Score A:{reba_result['score_A']}  B:{reba_result['score_B']}"
      f"  C:{reba_result['score_C']}  Act:+{reba_result['mod_actividad']}",
      10, y, C_TXT, 0.38)
    y += 16

    color_r = reba_result["color_riesgo"]
    cv2.rectangle(frame, (5, y - 4), (panel_w - 5, y + 36), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, y - 4), (panel_w - 5, y + 36), color_r, 2)
    # Score total de REBA
    t(f"REBA SCORE: {reba_result['score_reba']}/15", 12, y + 13, color_r, 0.6, 2)
    # Score final y nivel de riesgo (destacado)
    t(f"Riesgo: {reba_result['nivel_riesgo']}", 12, y + 29, color_r, 0.48)


def dibujar_angulo_en_articulacion(frame, punto_px_c, angulo, label, valido=True):
    """Dibuja el ángulo medido sobre la articulación correspondiente."""
    x, y = punto_px_c
    color = (255, 255, 100) if valido else (100, 100, 200)
    texto = f"{label}: {angulo:.0f}"
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(frame, (x + 5, y - th - 3), (x + tw + 9, y + 2), (0, 0, 0), -1)
    cv2.putText(frame, texto, (x + 7, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def dibujar_barra_buffer(frame, porcentaje, w, h):
    bh = 6
    by = h - bh - 2
    bw = int((w - 350) * porcentaje / 100)
    cv2.rectangle(frame, (345, by), (w - 5, by + bh), (50, 50, 50), -1)
    if bw > 0:
        cv2.rectangle(frame, (345, by), (345 + bw, by + bh), (50, 200, 150), -1)
    cv2.putText(frame, f"Buffer ACF: {porcentaje:.0f}%",
                (345, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (150, 200, 150), 1, cv2.LINE_AA)


def pedir_altura():
    """Solicita la altura del sujeto por consola con validación."""
    while True:
        try:
            h = float(input("\n[Calibración] Introduce la altura del sujeto en cm (ej: 175): "))
            if 100 <= h <= 230:
                return h
            print("Por favor introduce un valor entre 100 y 230 cm.")
        except ValueError:
            print("Valor no válido. Introduce un número.")


def guardar_fila_csv(writer, timestamp, lado, angulos_dict, reba_result, rep_resultado):
    """Escribe una fila de datos en el CSV."""
    writer.writerow([
        timestamp, lado,
        round(angulos_dict.get("tronco", 0), 2),
        round(angulos_dict.get("cuello", 0), 2),
        round(angulos_dict.get("brazo", 0), 2),
        round(angulos_dict.get("antebrazo", 0), 2),
        round(angulos_dict.get("muneca", 0), 2),
        round(angulos_dict.get("rodilla", 0), 2),
        reba_result["s_tronco"], reba_result["s_cuello"],
        reba_result["s_piernas"], reba_result["s_brazo"],
        reba_result["s_antebrazo"], reba_result["s_muneca"],
        reba_result["score_A"], reba_result["score_B"],
        reba_result["score_C"], reba_result["score_reba"],
        reba_result["nivel_riesgo"],
        int(rep_resultado["es_repetitivo"]),
        round(rep_resultado["frecuencia_rpm"], 2),
        round(rep_resultado["confianza"], 3),
    ])


# ==============================================================================
# PROGRAMA PRINCIPAL
# ==============================================================================

def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # --- Configuración de MediaPipe ---
    pose_config = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1, # 0=rápido, 1=equilibrado, 2=preciso
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # --- Cámara ---
    cap = cv2.VideoCapture(CAMARA_ID)
    if not cap.isOpened():
        print(f"[Error] No se pudo abrir la cámara {CAMARA_ID}")
        return

    # Intentar subir resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30

    # --- Calibrador ---
    calibrador = Calibrador()
    detector_rep = DetectorRepeticiones(ventana_s=VENTANA_AUTOCORRELACION_S, fps=fps_cam)
    filtro_ema = FiltroAngulos(alpha=ALPHA_EMA)
    gestor_oclusion = GestorOclusiones(umbral_visibilidad=UMBRAL_VISIBILIDAD)

    # Preguntar altura inicial
    print("\n=== SISTEMA DE ANÁLISIS ERGONÓMICO REBA ===")
    print("Controles: [ESPACIO] calibrar | [R] grabar | [S] captura | [ESC] salir")
    altura_inicial = pedir_altura()
    calibrador.iniciar_calibracion(altura_inicial)

    # --- CSV ---
    csv_file = None
    csv_writer = None
    if GUARDAR_CSV:
        nombre_csv = f"reba_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_file = open(nombre_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp", "lado",
            "ang_tronco", "ang_cuello", "ang_brazo",
            "ang_antebrazo", "ang_muneca", "ang_rodilla",
            "s_tronco", "s_cuello", "s_piernas",
            "s_brazo", "s_antebrazo", "s_muneca",
            "score_A", "score_B", "score_C",
            "score_reba", "nivel_riesgo",
            "repetitivo", "frecuencia_rpm", "confianza_acf",
        ])
        print(f"[CSV] Guardando datos en: {nombre_csv}")

    # --- Grabación de vídeo ---
    grabando = False
    video_writer = None
    # --- Bucle principal ---
    reba_result_ant = None
    angulos_ant = {}
    rep_resultado_ant = detector_rep.resultado
    frame_count = 0
    t_inicio = time.time()

    with pose_config as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] No se recibe imagen de la cámara.")
                break

            h_f, w_f, _ = frame.shape
            frame_count += 1
            # FPS real calculado cada 30 frames para evitar fluctuaciones
            fps_real = frame_count / max(time.time() - t_inicio, 0.001)

            # --- PROCESAR CON MEDIAPIPE ---
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True

            reba_result = reba_result_ant
            angulos_dict = angulos_ant
            rep_resultado = rep_resultado_ant
            lado = "izq"
            validez = {}

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Calibración en progreso
                if calibrador.calibrando:
                    calibrador.actualizar(landmarks, h_f)

                # Seleccionar lado más visible y extraer puntos clave
                lado = seleccionar_lado(landmarks)
                puntos = extraer_landmarks(landmarks, lado)

                # Calcula ángulos
                ang_tronco_raw   = angulo_tronco_vertical(puntos["hombro"], puntos["cadera"]) # TRONCO: ángulo del segmento cadera-hombro respecto a la vertical
                ang_cuello_raw   = angulo_cuello(puntos["oreja"], puntos["hombro"], puntos["cadera"]) # CUELLO: ángulo de la cabeza respecto al tronco
                ang_brazo_raw    = angulo_brazo_tronco(puntos["hombro"], puntos["codo"], puntos["cadera"]) # BRAZO: ángulo del húmero respecto al tronco
                ang_antebrazo_raw= angulo_antebrazo(puntos["hombro"], puntos["codo"], puntos["muneca"]) # ANTEBRAZO: ángulo del antebrazo respecto al brazo, codo como vértice (hombro-codo-muñeca)
                ang_muneca_raw   = angulo_muneca(puntos["codo"], puntos["muneca"], puntos["indice"]) # MUNECA: ángulo de la muñeca respecto al antebrazo (codo-muñeca-índice)
                ang_rodilla_raw  = calcular_angulo_rodilla_reba(puntos) # RODILLA: ángulo de la rodilla respecto a la vertical (flexión)

                # Gestión de oclusiones
                ang_tronco,    val_t = gestor_oclusion.validar_angulo("tronco",    ang_tronco_raw,    [puntos["hombro_vis"], puntos["cadera_vis"]])
                ang_cuello,    val_c = gestor_oclusion.validar_angulo("cuello",    ang_cuello_raw,    [puntos["oreja_vis"],  puntos["hombro_vis"]])
                ang_brazo,     val_b = gestor_oclusion.validar_angulo("brazo",     ang_brazo_raw,     [puntos["hombro_vis"], puntos["codo_vis"]])
                ang_antebrazo, val_a = gestor_oclusion.validar_angulo("antebrazo", ang_antebrazo_raw, [puntos["codo_vis"],   puntos["muneca_vis"]])
                ang_muneca,    val_m = gestor_oclusion.validar_angulo("muneca",    ang_muneca_raw,    [puntos["muneca_vis"], puntos["indice_vis"]])
                ang_rodilla,   val_r = gestor_oclusion.validar_angulo("rodilla",   ang_rodilla_raw,   [puntos["rodilla_vis"],puntos["tobillo_vis"]])

                validez = {"tronco": val_t, "cuello": val_c, "brazo": val_b,
                           "antebrazo": val_a, "muneca": val_m, "rodilla": val_r}

                # Filtro EMA
                angulos_dict = filtro_ema.filtrar_dict({
                    "tronco": ang_tronco, "cuello": ang_cuello,
                    "brazo": ang_brazo, "antebrazo": ang_antebrazo,
                    "muneca": ang_muneca, "rodilla": ang_rodilla,
                })

                # Detector de repeticiones
                detector_rep.añadir_muestra([
                    angulos_dict["tronco"], angulos_dict["cuello"],
                    angulos_dict["brazo"], angulos_dict["antebrazo"],
                ])
                rep_resultado = detector_rep.analizar(frame_count)
                rep_resultado_ant = rep_resultado

                # SCORE REBA con modificador automático
                reba_result = calcular_reba_completo(
                    angulo_tronco=angulos_dict["tronco"],
                    angulo_cuello=angulos_dict["cuello"],
                    angulo_rodilla=angulos_dict["rodilla"],
                    angulo_brazo=angulos_dict["brazo"],
                    angulo_antebrazo_val=angulos_dict["antebrazo"],
                    angulo_muneca_val=angulos_dict["muneca"],
                    soporte_bilateral=True, # Ajustable según experimento
                    actividad_repetitiva=rep_resultado["es_repetitivo"],
                )

                reba_result_ant = reba_result
                angulos_ant = angulos_dict

                # Guardar en CSV cada 15 frames (~0.5s a 30fps)
                if GUARDAR_CSV and csv_writer and frame_count % 15 == 0:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    guardar_fila_csv(csv_writer, ts, lado, angulos_dict, reba_result, rep_resultado)

                # Dibujar esqueleto
                if MOSTRAR_ESQUELETO:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                # Dibujar ángulos
                def px(n): return punto_px(puntos[n], w_f, h_f)
                dibujar_angulo_en_articulacion(frame, px("cadera"),  angulos_dict["tronco"],    "Tronco",    validez.get("tronco", True))
                dibujar_angulo_en_articulacion(frame, px("hombro"),  angulos_dict["cuello"],    "Cuello",    validez.get("cuello", True))
                dibujar_angulo_en_articulacion(frame, px("codo"),    angulos_dict["brazo"],     "Brazo",     validez.get("brazo", True))
                dibujar_angulo_en_articulacion(frame, px("muneca"),  angulos_dict["antebrazo"], "Antebrazo", validez.get("antebrazo", True))
                dibujar_angulo_en_articulacion(frame, px("rodilla"), angulos_dict["rodilla"],   "Rodilla",   validez.get("rodilla", True))

            # Dibujar panel de información
            if reba_result:
                dibujar_panel_info(frame, reba_result, angulos_dict, lado,
                                   calibrador, rep_resultado, w_f, h_f)
            else:
                # Si no se detectan landmarks, mostrar mensaje central
                cv2.putText(frame, "Sin deteccion de persona",
                            (360, h_f // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # Si el calibrador está en modo de calibración, mostrar mensaje
            if calibrador.calibrando:
                n = len(calibrador.muestras_altura_px)
                cv2.putText(frame,
                            f"CALIBRANDO... {n}/{calibrador.n_muestras} - Mantente erguido",
                            (360, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 220, 255), 2, cv2.LINE_AA)

            dibujar_barra_buffer(frame, detector_rep.porcentaje_buffer(), w_f, h_f)

            # Mostrar FPS real en la esquina superior derecha
            cv2.putText(frame, f"FPS: {fps_real:.1f}", (w_f - 110, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Mostrar indicador de grabación
            if grabando:
                cv2.circle(frame, (w_f - 20, 55), 8, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (w_f - 70, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            cv2.imshow("Análisis REBA v2 — Ergonomía", frame)
            # Grabación de vídeo
            if grabando and video_writer:
                video_writer.write(frame)

            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC para salir
                break
            elif key == ord(' '): # ESPACIO para calibrar
                nueva = pedir_altura()
                calibrador.iniciar_calibracion(nueva)
                detector_rep.__init__(VENTANA_AUTOCORRELACION_S, fps_cam)
                filtro_ema.__init__(ALPHA_EMA)
            elif key in (ord('r'), ord('R')): # R para grabar
                if not grabando:
                    nombre_v = f"reba_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(nombre_v, fourcc, fps_cam, (w_f, h_f))
                    grabando = True
                    print(f"[REC] → {nombre_v}")
                else:
                    grabando = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    print("[REC] Detenida.")
            elif key in (ord('s'), ord('S')): # S para captura de pantalla
                nombre_c = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(nombre_c, frame)
                print(f"[Captura] → {nombre_c}")

    # Limpieza
    cap.release()
    if video_writer:
        video_writer.release()
    if csv_file:
        csv_file.close()
    cv2.destroyAllWindows()
    print("[Sistema] Finalizado.")


if __name__ == "__main__":
    main()

""" 
  - Ángulos correctos respecto a vertical/horizontal 
  - Calibración por altura del sujeto
  - Selección automática del lado más visible
  - Score REBA completo con tablas oficiales
  - Visualización con overlay de puntuaciones
  - Grabación de sesión opcional
  - Exportación de datos a CSV
  - Detección automática de movimientos repetitivos (autocorrelación)
  - Filtro EMA para suavizar ángulos ruidosos (especialmente muñeca/mano)
  - Gestión de oclusiones: usa último valor válido si landmark no visible
  - El modificador de actividad REBA se calcula automáticamente
  - Panel actualizado con estado de repetición y buffer de autocorrelación

Controles:
  [ESPACIO] - Iniciar/reiniciar calibración
  [R]       - Iniciar/detener grabación
  [S]       - Guardar captura de pantalla
  [ESC]     - Salir
"""