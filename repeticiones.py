import numpy as np
from collections import deque


class DetectorRepeticiones:
    """
    Analiza un buffer de ángulos articulares para detectar movimientos repetitivos
    mediante autocorrelación temporal.
    """

    def __init__(self, ventana_s=10.0, fps=30, umbral_repeticion=0.35):
        """
        ventana_s: segundos de buffer a analizar (>= 5 recomendado)
        fps: frames por segundo de la cámara
        umbral_repeticion: distancia normalizada máxima para considerar repetición
                           (0 = idéntico, 1 = máxima diferencia; 0.35 funciona bien)
        """
        self.fps = fps
        self.ventana_frames = int(ventana_s * fps)
        self.umbral = umbral_repeticion

        # Buffer circular de muestras. Cada muestra es un array de ángulos.
        self.buffer = deque(maxlen=self.ventana_frames)

        # Resultado del último análisis
        self.resultado = {
            "es_repetitivo": False,
            "frecuencia_rpm": 0.0,
            "periodo_frames": 0,
            "confianza": 0.0,
            "modificador_rula": 0,
            "modificador_reba": 0,
        }

        # Cuándo se hizo el último análisis (para no analizar cada frame)
        self._ultimo_analisis_frame = 0
        self._intervalo_analisis = max(10, int(fps * 0.5))  # Cada 0.5s

    def añadir_muestra(self, angulos: list):
        """
        Añade una muestra de ángulos al buffer.
        angulos: lista de floats con los ángulos articulares del frame actual.
        """
        self.buffer.append(np.array(angulos, dtype=np.float32))

    def analizar(self, frame_actual: int = 0) -> dict:
        """
        Analiza el buffer actual y devuelve si hay movimiento repetitivo.
        Solo recalcula cada self._intervalo_analisis frames para eficiencia.
        """
        # Throttle: no calcular cada frame
        if (frame_actual - self._ultimo_analisis_frame < self._intervalo_analisis
                and frame_actual > 0):
            return self.resultado

        self._ultimo_analisis_frame = frame_actual

        n = len(self.buffer)
        # Necesitamos al menos 3 segundos de datos
        if n < self.fps * 3:
            return self.resultado

        # Convertir buffer a matriz (n_frames x n_angulos)
        matriz = np.array(self.buffer)  # shape: (n, n_angulos)

        # Normalizar cada columna (ángulo) por su rango para evitar que
        # ángulos grandes (tronco) dominen sobre pequeños (muñeca)
        rangos = np.ptp(matriz, axis=0)  # Peak-to-peak por columna
        rangos[rangos < 1.0] = 1.0       # Evitar división por cero
        matriz_norm = (matriz - matriz.min(axis=0)) / rangos

        # Calcular la señal univariada como norma del vector de ángulos
        # (reduce la dimensionalidad manteniendo la dinámica del movimiento)
        señal = np.linalg.norm(matriz_norm, axis=1)

        # Autocorrelación: comparar la señal con versiones desplazadas de sí misma
        resultado_acf = self._autocorrelacion(señal)

        self.resultado = resultado_acf
        return self.resultado

    def _autocorrelacion(self, señal: np.ndarray) -> dict:
        """
        Calcula la autocorrelación de la señal y detecta periodicidad.

        La autocorrelación mide cuánto se parece la señal a sí misma
        desplazada T pasos en el tiempo. Un pico en el lag T indica que
        el movimiento se repite cada T frames.
        """
        n = len(señal)
        señal_c = señal - señal.mean()  # Centrar

        # ACF normalizada
        acf_full = np.correlate(señal_c, señal_c, mode='full')
        acf = acf_full[n - 1:]  # Solo lags >= 0
        if acf[0] == 0:
            return self._sin_repeticion()
        acf = acf / acf[0]  # Normalizar a [0, 1]

        # Buscar picos en el rango de periodos razonables:
        # Mínimo: 0.5s (120 rpm), Máximo: 5s (12 rpm)
        lag_min = max(int(self.fps * 0.5), 5)
        lag_max = min(int(self.fps * 5.0), n // 2)

        if lag_min >= lag_max:
            return self._sin_repeticion()

        acf_rango = acf[lag_min:lag_max]

        # Encontrar el máximo pico en el rango
        idx_pico = np.argmax(acf_rango)
        valor_pico = acf_rango[idx_pico]
        periodo_frames = idx_pico + lag_min

        # El movimiento es repetitivo si la autocorrelación supera el umbral
        es_repetitivo = valor_pico > (1.0 - self.umbral)

        if es_repetitivo:
            frecuencia_rpm = (self.fps / periodo_frames) * 60.0
            # Modificador RULA: +1 si > 4 repeticiones/min, es decir periodo < 15s
            # En la práctica, cualquier repetición detectada suma +1
            mod_rula = 1
            # REBA usa el mismo concepto
            mod_reba = 1
        else:
            frecuencia_rpm = 0.0
            mod_rula = 0
            mod_reba = 0

        return {
            "es_repetitivo": es_repetitivo,
            "frecuencia_rpm": frecuencia_rpm,
            "periodo_frames": periodo_frames,
            "confianza": float(valor_pico),
            "modificador_rula": mod_rula,
            "modificador_reba": mod_reba,
        }

    def _sin_repeticion(self) -> dict:
        return {
            "es_repetitivo": False,
            "frecuencia_rpm": 0.0,
            "periodo_frames": 0,
            "confianza": 0.0,
            "modificador_rula": 0,
            "modificador_reba": 0,
        }

    def porcentaje_buffer(self) -> float:
        """Porcentaje del buffer lleno (0-100). Útil para la UI."""
        return (len(self.buffer) / self.ventana_frames) * 100.0


class FiltroAngulos:
    """
    Filtro de media móvil exponencial (EMA) para suavizar ángulos ruidosos.

    Aplicado especialmente a muñeca y mano, donde MediaPipe es menos preciso
    (problema señalado en el artículo para keypoints distales).
    """

    def __init__(self, alpha: float = 0.3):
        """
        alpha: factor de suavizado.
          - Cerca de 1.0 → sin suavizado (responde rápido, ruidoso)
          - Cerca de 0.0 → mucho suavizado (responde lento, estable)
          - 0.3 es un buen equilibrio para ergonomía
        """
        self.alpha = alpha
        self._valores = {}

    def filtrar(self, nombre: str, valor: float) -> float:
        """
        Aplica el filtro EMA a un ángulo identificado por 'nombre'.
        Devuelve el valor suavizado.
        """
        if nombre not in self._valores:
            self._valores[nombre] = valor
        else:
            self._valores[nombre] = (self.alpha * valor
                                     + (1 - self.alpha) * self._valores[nombre])
        return self._valores[nombre]

    def filtrar_dict(self, angulos: dict) -> dict:
        """Aplica el filtro a todos los ángulos de un diccionario."""
        return {k: self.filtrar(k, v) for k, v in angulos.items()}


class GestorOclusiones:
    """
    Gestiona los casos en que un landmark tiene baja visibilidad (oclusión).

    Estrategia: si la visibilidad cae por debajo del umbral, usar el último
    valor válido del ángulo en lugar de calcular uno potencialmente erróneo.
    Esto evita saltos bruscos en el score REBA causados por keypoints incorrectos.
    """

    def __init__(self, umbral_visibilidad: float = 0.5, max_frames_stale: int = 15):
        """
        umbral_visibilidad: visibilidad mínima para considerar un punto válido (0-1).
        max_frames_stale: máximo de frames que se puede usar el último valor válido.
        """
        self.umbral = umbral_visibilidad
        self.max_stale = max_frames_stale
        self._ultimos_angulos = {}
        self._frames_stale = {}

    def validar_angulo(self, nombre: str, angulo: float,
                       visibilidades: list) -> tuple:
        """
        Decide si usar el ángulo calculado o el último válido.

        nombre: identificador del ángulo (ej: "tronco")
        angulo: valor calculado en este frame
        visibilidades: lista de visibilidades de los landmarks usados [0, 1]

        Retorna: (angulo_final, es_valido)
        """
        vis_min = min(visibilidades) if visibilidades else 0.0

        if vis_min >= self.umbral:
            # Landmark visible: actualizar y usar valor actual
            self._ultimos_angulos[nombre] = angulo
            self._frames_stale[nombre] = 0
            return angulo, True
        else:
            # Landmark oculto: usar último valor si no es muy antiguo
            stale = self._frames_stale.get(nombre, self.max_stale + 1)
            if stale <= self.max_stale and nombre in self._ultimos_angulos:
                self._frames_stale[nombre] = stale + 1
                return self._ultimos_angulos[nombre], False
            else:
                # Sin valor anterior o demasiado antiguo: usar valor actual
                # aunque sea impreciso
                self._frames_stale[nombre] = self.max_stale + 1
                return angulo, False

"""
Detección de movimientos repetitivos por autocorrelación.

Basado en el método descrito en:
  "Automated Assessment of Assembly Work Ergonomics Using RULA"
  (ver artículo de referencia)

Idea:
  - Se guarda un buffer temporal de vectores de ángulos (uno por frame).
  - Se calcula la autocorrelación del buffer: se compara cada ventana
    temporal con ventanas desplazadas en el tiempo.
  - Si la distancia entre ventanas cae periódicamente, hay movimiento repetitivo.
  - El periodo detectado determina la frecuencia de repetición.

Uso:
  detector = DetectorRepeticiones(ventana_s=10, fps=30)
  detector.añadir_muestra([ang_tronco, ang_cuello, ang_brazo, ...])
  resultado = detector.analizar()
  # resultado['es_repetitivo'] -> bool
  # resultado['frecuencia_rpm'] -> repeticiones por minuto
  # resultado['modificador_rula'] -> +1 si repetitivo
"""