import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
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


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

ALTURA_REFERENCIA_CM = 170  # Altura de referencia estándar en cm
CAMARA_ID = 0               # ID de la cámara (0 = webcam por defecto)
GUARDAR_CSV = True          # Guardar datos en CSV automáticamente
MOSTRAR_ESQUELETO = True     # Mostrar landmarks de MediaPipe


# ==============================================================================
# CLASE DE CALIBRACIÓN
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
        self.factor_escala = 1.0  # Ratio altura_sujeto / altura_referencia
        self.muestras_altura_px = []
        self.n_muestras = 30  # Promedia 30 frames para más estabilidad
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
            print(f"[Calibración] Error en muestra: {e}")

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
        print(f"  Factor escala: {self.factor_escala:.3f}")

    def estado(self):
        """Devuelve el estado actual de la calibración como string."""
        if self.calibrado:
            return f"CAL OK ({self.altura_sujeto_cm}cm)"
        elif self.calibrando:
            progreso = len(self.muestras_altura_px)
            return f"Calibrando... {progreso}/{self.n_muestras}"
        else:
            return "Sin calibrar [ESPACIO]"

    def segundos_calibrando(self):
        if self.inicio_calibracion:
            return time.time() - self.inicio_calibracion
        return 0


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def seleccionar_lado(landmarks):
    """
    Selecciona el lado (izquierdo/derecho) más visible para el análisis.
    Compara la visibilidad media de hombro, codo, muñeca, cadera, rodilla.
    """
    indices_izq = [11, 13, 15, 23, 25, 27]  # hombro, codo, muñeca, cadera, rodilla, tobillo
    indices_der = [12, 14, 16, 24, 26, 28]

    vis_izq = np.mean([landmarks[i].visibility for i in indices_izq])
    vis_der = np.mean([landmarks[i].visibility for i in indices_der])

    return "izq" if vis_izq >= vis_der else "der"


def obtener_punto(landmarks, idx, w, h, normalizado=True):
    """
    Devuelve las coordenadas de un landmark.
    
    normalizado=True → [x_norm, y_norm] (0 a 1)
    normalizado=False → [x_px, y_px] (píxeles)
    """
    lm = landmarks[idx]
    if normalizado:
        return [lm.x, lm.y]
    else:
        return [int(lm.x * w), int(lm.y * h)]


def punto_px(punto_norm, w, h):
    """Convierte coordenadas normalizadas a píxeles."""
    return (int(punto_norm[0] * w), int(punto_norm[1] * h))


def extraer_landmarks(landmarks, lado):
    """
    Extrae los landmarks relevantes según el lado seleccionado.
    Devuelve un diccionario con coordenadas normalizadas [x, y].
    """
    if lado == "izq":
        idx = {
            "hombro": 11, "codo": 13, "muneca": 15,
            "indice": 19,  # Índice de la mano
            "cadera": 23, "rodilla": 25, "tobillo": 27,
            "oreja": 7,
        }
    else:
        idx = {
            "hombro": 12, "codo": 14, "muneca": 16,
            "indice": 20,
            "cadera": 24, "rodilla": 26, "tobillo": 28,
            "oreja": 8,
        }
    # Nariz es siempre 0
    idx["nariz"] = 0

    puntos = {}
    for nombre, i in idx.items():
        lm = landmarks[i]
        puntos[nombre] = [lm.x, lm.y]
        puntos[nombre + "_vis"] = lm.visibility

    return puntos


def calcular_angulo_rodilla_reba(puntos):
    """
    Calcula la flexión de la rodilla para REBA.
    0° = pierna recta, valores positivos = flexión.
    """
    ang_3p = calcular_angulo_3puntos(
        puntos["cadera"], puntos["rodilla"], puntos["tobillo"]
    )
    # El ángulo 3 puntos da 180° cuando la pierna está recta
    return abs(180 - ang_3p)


def dibujar_panel_info(frame, reba_result, angulos_dict, lado, calibrador, w, h):
    """
    Dibuja el panel de información ergonómica en el frame.
    """
    # Panel de fondo semitransparente en la parte izquierda
    overlay = frame.copy()
    panel_w = 320
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    color_titulo = (200, 200, 255)
    color_ok = (100, 255, 100)
    color_warn = (50, 200, 255)
    color_alto = (50, 100, 255)
    color_texto = (220, 220, 220)

    def texto(msg, x, y, color=(220, 220, 220), escala=0.5, grosor=1):
        cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    escala, color, grosor, cv2.LINE_AA)

    y = 22
    # Título
    texto("ANÁLISIS REBA", 10, y, color_titulo, 0.65, 2)
    y += 6
    cv2.line(frame, (10, y), (panel_w - 10, y), color_titulo, 1)
    y += 16

    # Estado calibración
    estado_cal = calibrador.estado()
    col_cal = color_ok if calibrador.calibrado else (50, 150, 255)
    texto(f"Calibracion: {estado_cal}", 10, y, col_cal, 0.45, 1)
    y += 14
    texto(f"Lado analizado: {lado.upper()}", 10, y, color_texto, 0.45)
    y += 18

    cv2.line(frame, (10, y), (panel_w - 10, y), (80, 80, 80), 1)
    y += 12

    # Ángulos
    texto("ÁNGULOS (grados)", 10, y, color_titulo, 0.48, 1)
    y += 14
    ang_data = [
        ("Tronco/vertical", angulos_dict.get("tronco", 0)),
        ("Cuello/tronco", angulos_dict.get("cuello", 0)),
        ("Brazo/tronco", angulos_dict.get("brazo", 0)),
        ("Antebrazo (codo)", angulos_dict.get("antebrazo", 0)),
        ("Muñeca", angulos_dict.get("muneca", 0)),
        ("Rodilla (flex.)", angulos_dict.get("rodilla", 0)),
    ]
    for nombre, valor in ang_data:
        texto(f"  {nombre}: {valor:.1f}°", 10, y, color_texto, 0.42)
        y += 13

    y += 5
    cv2.line(frame, (10, y), (panel_w - 10, y), (80, 80, 80), 1)
    y += 12

    # Scores individuales
    texto("SCORES INDIVIDUALES", 10, y, color_titulo, 0.48, 1)
    y += 14

    score_items = [
        ("Tronco", reba_result["s_tronco"], 4),
        ("Cuello", reba_result["s_cuello"], 3),
        ("Piernas", reba_result["s_piernas"], 4),
        ("Brazo", reba_result["s_brazo"], 6),
        ("Antebrazo", reba_result["s_antebrazo"], 2),
        ("Muñeca", reba_result["s_muneca"], 3),
    ]

    for nombre, score, max_score in score_items:
        # Color según nivel del score
        ratio = score / max_score
        if ratio <= 0.4:
            sc = color_ok
        elif ratio <= 0.7:
            sc = color_warn
        else:
            sc = color_alto

        # Barra de progreso
        bar_x = 170
        bar_w = 100
        bar_h = 8
        bar_filled = int(bar_w * ratio)
        cv2.rectangle(frame, (bar_x, y - 8), (bar_x + bar_w, y), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, y - 8), (bar_x + bar_filled, y), sc, -1)

        texto(f"  {nombre}: {score}/{max_score}", 10, y, sc, 0.42)
        y += 13

    y += 5
    cv2.line(frame, (10, y), (panel_w - 10, y), (80, 80, 80), 1)
    y += 12

    # Scores de tabla
    texto("PUNTUACIÓN REBA", 10, y, color_titulo, 0.5, 1)
    y += 14
    texto(f"  Score A (Grupo A + carga): {reba_result['score_A']}", 10, y, color_texto, 0.43)
    y += 13
    texto(f"  Score B (Grupo B + agarre): {reba_result['score_B']}", 10, y, color_texto, 0.43)
    y += 13
    texto(f"  Score C (Tabla C): {reba_result['score_C']}", 10, y, color_texto, 0.43)
    y += 13
    texto(f"  Mod. actividad: +{reba_result['mod_actividad']}", 10, y, color_texto, 0.43)
    y += 18

    # Score final y nivel de riesgo (destacado)
    score_final = reba_result["score_reba"]
    nivel = reba_result["nivel_riesgo"]
    color_riesgo = reba_result["color_riesgo"]

    cv2.rectangle(frame, (5, y - 4), (panel_w - 5, y + 35), color_riesgo, 2)
    cv2.rectangle(frame, (5, y - 4), (panel_w - 5, y + 35), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, y - 4), (panel_w - 5, y + 35), color_riesgo, 2)

    texto(f"REBA SCORE: {score_final}/15", 12, y + 12, color_riesgo, 0.6, 2)
    texto(f"Riesgo: {nivel}", 12, y + 28, color_riesgo, 0.5, 1)

    y += 50
    texto(reba_result["accion"], 10, y, (200, 200, 200), 0.38)


def dibujar_angulo_en_articulacion(frame, punto_px_centro, angulo, label, w, h):
    """Dibuja el ángulo medido sobre la articulación correspondiente."""
    x, y = punto_px_centro
    # Fondo pequeño
    texto = f"{label}: {angulo:.0f}"
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(frame, (x + 5, y - th - 4), (x + tw + 10, y + 2), (0, 0, 0), -1)
    cv2.putText(frame, texto, (x + 7, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1, cv2.LINE_AA)


def pedir_altura():
    """Solicita la altura del sujeto por consola con validación."""
    while True:
        try:
            h = float(input("\n[Calibración] Introduce la altura del sujeto en cm (ej: 175): "))
            if 100 <= h <= 230:
                return h
            else:
                print("Por favor introduce un valor entre 100 y 230 cm.")
        except ValueError:
            print("Valor no válido. Introduce un número.")


def guardar_fila_csv(writer, timestamp, lado, angulos_dict, reba_result):
    """Escribe una fila de datos en el CSV."""
    writer.writerow([
        timestamp,
        lado,
        round(angulos_dict.get("tronco", 0), 2),
        round(angulos_dict.get("cuello", 0), 2),
        round(angulos_dict.get("brazo", 0), 2),
        round(angulos_dict.get("antebrazo", 0), 2),
        round(angulos_dict.get("muneca", 0), 2),
        round(angulos_dict.get("rodilla", 0), 2),
        reba_result["s_tronco"],
        reba_result["s_cuello"],
        reba_result["s_piernas"],
        reba_result["s_brazo"],
        reba_result["s_antebrazo"],
        reba_result["s_muneca"],
        reba_result["score_A"],
        reba_result["score_B"],
        reba_result["score_C"],
        reba_result["score_reba"],
        reba_result["nivel_riesgo"],
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
        model_complexity=1,           # 0=rápido, 1=equilibrado, 2=preciso
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
    print(f"[Cámara] Resolución: {w}x{h} @ {fps_cam:.0f} fps")

    # --- Calibrador ---
    calibrador = Calibrador()

    # Preguntar altura inicial
    print("\n=== SISTEMA DE ANÁLISIS ERGONÓMICO REBA ===")
    print("Controles:")
    print("  [ESPACIO] - Calibrar con nueva altura")
    print("  [R]       - Iniciar/detener grabación")
    print("  [S]       - Captura de pantalla")
    print("  [ESC]     - Salir")

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
        ])
        print(f"[CSV] Guardando datos en: {nombre_csv}")

    # --- Grabación de vídeo ---
    grabando = False
    video_writer = None

    # --- Bucle principal ---
    reba_result_anterior = None
    angulos_anterior = {}
    frame_count = 0
    t_inicio = time.time()

    with pose_config as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] No se recibe imagen de la cámara.")
                break

            h_frame, w_frame, _ = frame.shape
            frame_count += 1

            # FPS real
            elapsed = time.time() - t_inicio
            fps_real = frame_count / elapsed if elapsed > 0 else 0

            # --- PROCESAR CON MEDIAPIPE ---
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True

            reba_result = reba_result_anterior
            angulos_dict = angulos_anterior
            lado = "izq"

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Calibración en progreso
                if calibrador.calibrando:
                    calibrador.actualizar(landmarks, h_frame)

                # Seleccionar lado más visible
                lado = seleccionar_lado(landmarks)
                puntos = extraer_landmarks(landmarks, lado)

                # ==============================
                # CALCULAR ÁNGULOS CORRECTAMENTE
                # ==============================

                # TRONCO: ángulo del segmento cadera-hombro respecto a la vertical
                ang_tronco = angulo_tronco_vertical(
                    puntos["hombro"], puntos["cadera"]
                )

                # CUELLO: ángulo de la cabeza respecto al tronco
                ang_cuello = angulo_cuello(
                    puntos["oreja"], puntos["hombro"], puntos["cadera"]
                )

                # BRAZO: ángulo del húmero respecto al tronco
                ang_brazo = angulo_brazo_tronco(
                    puntos["hombro"], puntos["codo"], puntos["cadera"]
                )

                # ANTEBRAZO: ángulo en el codo (hombro-codo-muñeca)
                ang_antebrazo = angulo_antebrazo(
                    puntos["hombro"], puntos["codo"], puntos["muneca"]
                )

                # MUÑECA: ángulo codo-muñeca-índice
                ang_muneca = angulo_muneca(
                    puntos["codo"], puntos["muneca"], puntos["indice"]
                )

                # RODILLA: flexión de la rodilla
                ang_rodilla = calcular_angulo_rodilla_reba(puntos)

                angulos_dict = {
                    "tronco": ang_tronco,
                    "cuello": ang_cuello,
                    "brazo": ang_brazo,
                    "antebrazo": ang_antebrazo,
                    "muneca": ang_muneca,
                    "rodilla": ang_rodilla,
                }

                # ==============================
                # SCORE REBA COMPLETO
                # ==============================
                reba_result = calcular_reba_completo(
                    angulo_tronco=ang_tronco,
                    angulo_cuello=ang_cuello,
                    angulo_rodilla=ang_rodilla,
                    angulo_brazo=ang_brazo,
                    angulo_antebrazo_val=ang_antebrazo,
                    angulo_muneca_val=ang_muneca,
                    soporte_bilateral=True,  # Ajustable según experimento
                )

                reba_result_anterior = reba_result
                angulos_anterior = angulos_dict

                # Guardar en CSV cada 15 frames (~0.5s a 30fps)
                if GUARDAR_CSV and csv_writer and frame_count % 15 == 0:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    guardar_fila_csv(csv_writer, ts, lado, angulos_dict, reba_result)

                # ==============================
                # DIBUJAR ESQUELETO
                # ==============================
                if MOSTRAR_ESQUELETO:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

                # ==============================
                # DIBUJAR ÁNGULOS SOBRE CUERPO
                # ==============================
                def px(nombre):
                    return punto_px(puntos[nombre], w_frame, h_frame)

                dibujar_angulo_en_articulacion(frame, px("cadera"), ang_tronco, "Tronco", w_frame, h_frame)
                dibujar_angulo_en_articulacion(frame, px("hombro"), ang_cuello, "Cuello", w_frame, h_frame)
                dibujar_angulo_en_articulacion(frame, px("codo"), ang_brazo, "Brazo", w_frame, h_frame)
                dibujar_angulo_en_articulacion(frame, px("muneca"), ang_antebrazo, "Antebrazo", w_frame, h_frame)
                dibujar_angulo_en_articulacion(frame, px("rodilla"), ang_rodilla, "Rodilla", w_frame, h_frame)

            # ==============================
            # PANEL DE INFORMACIÓN
            # ==============================
            if reba_result:
                dibujar_panel_info(frame, reba_result, angulos_dict, lado, calibrador, w_frame, h_frame)
            else:
                # Sin detección
                cv2.putText(frame, "Sin deteccion de persona", (340, h_frame // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # Mensaje calibración en curso
            if calibrador.calibrando:
                progreso = len(calibrador.muestras_altura_px)
                msg = f"CALIBRANDO... {progreso}/{calibrador.n_muestras} - Mantente de pie erguido"
                cv2.putText(frame, msg, (340, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

            # FPS en esquina superior derecha
            cv2.putText(frame, f"FPS: {fps_real:.1f}", (w_frame - 110, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # Indicador de grabación
            if grabando:
                cv2.circle(frame, (w_frame - 20, 60), 8, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (w_frame - 70, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # ==============================
            # MOSTRAR VENTANA
            # ==============================
            cv2.imshow("Análisis REBA - Ergonomía", frame)

            # Grabación de vídeo
            if grabando and video_writer:
                video_writer.write(frame)

            # ==============================
            # CONTROLES DE TECLADO
            # ==============================
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC - Salir
                break

            elif key == ord(' '):  # ESPACIO - Recalibrar
                nueva_altura = pedir_altura()
                calibrador.iniciar_calibracion(nueva_altura)

            elif key == ord('r') or key == ord('R'):  # R - Grabar
                if not grabando:
                    nombre_video = f"reba_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(nombre_video, fourcc, fps_cam, (w_frame, h_frame))
                    grabando = True
                    print(f"[Grabación] Iniciada: {nombre_video}")
                else:
                    grabando = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    print("[Grabación] Detenida.")

            elif key == ord('s') or key == ord('S'):  # S - Captura
                nombre_cap = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(nombre_cap, frame)
                print(f"[Captura] Guardada: {nombre_cap}")

    # Limpieza
    cap.release()
    if video_writer:
        video_writer.release()
    if csv_file:
        csv_file.close()
        print(f"[CSV] Archivo cerrado.")
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

Controles:
  [ESPACIO] - Iniciar/reiniciar calibración
  [R]       - Iniciar/detener grabación
  [S]       - Guardar captura de pantalla
  [ESC]     - Salir
"""