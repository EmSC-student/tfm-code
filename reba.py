# ==============================================================================
# GRUPO A: Tronco, Cuello, Piernas
# ==============================================================================

def puntuacion_tronco(angulo_grados, torsion=False, inclinacion_lateral=False):
    """
    Puntuación del tronco según REBA.
    
    angulo_grados: ángulo de flexión/extensión del tronco respecto a la vertical.
      - 0°       = erguido
      - positivo = flexión (hacia delante)
      - negativo = extensión (hacia atrás)
    
    torsion: True si hay rotación del tronco.
    inclinacion_lateral: True si hay inclinación lateral.
    
    Modificadores: +1 si hay torsión O inclinación lateral.
    """
    ang = abs(angulo_grados)

    if ang < 5:
        score = 1      # Erguido
    elif ang < 20:
        score = 2      # Flexión/extensión ligera
    elif ang < 60:
        score = 3      # Flexión moderada
    else:
        score = 4      # Flexión severa

    # Extensión (hacia atrás) también puntúa 2 si < 5°
    if angulo_grados < 0 and ang >= 5:  # Extensión > 5°
        score = max(score, 2)

    if torsion or inclinacion_lateral:
        score += 1

    return score


def puntuacion_cuello(angulo_grados, torsion=False, inclinacion_lateral=False):
    """
    Puntuación del cuello según REBA.
    
    angulo_grados: ángulo de flexión del cuello respecto al tronco.
      - 0-20°  = posición aceptable
      - >20°   = flexión excesiva
      - <0°    = extensión
    """
    ang = abs(angulo_grados)

    if angulo_grados < 0:  # Extensión
        score = 2
    elif ang <= 20:
        score = 1
    else:
        score = 2

    if torsion or inclinacion_lateral:
        score += 1

    return score


def puntuacion_piernas(angulo_rodilla_grados, soporte_bilateral=True):
    """
    Puntuación de las piernas según REBA.
    
    angulo_rodilla_grados: flexión de la rodilla.
    soporte_bilateral: True = apoyo bilateral/caminando, False = apoyo unilateral.
    
    Base:
      - Soporte bilateral/caminando = 1
      - Soporte unilateral/inestable = 2
    Modificadores por flexión de rodilla:
      - 30-60°: +1
      - >60°  : +2
    """
    if soporte_bilateral:
        score = 1
    else:
        score = 2

    ang = abs(angulo_rodilla_grados)
    if ang >= 60:
        score += 2
    elif ang >= 30:
        score += 1

    return score


# ==============================================================================
# GRUPO B: Brazo, Antebrazo, Muñeca
# ==============================================================================

def puntuacion_brazo(angulo_grados, abduccion=False, hombro_elevado=False, apoyo_brazo=False):
    """
    Puntuación del brazo superior (húmero) según REBA.
    
    angulo_grados: ángulo del brazo respecto al tronco (flexión/extensión).
      - 0°       = brazo pegado al cuerpo
      - positivo = elevación hacia delante
      - negativo = extensión hacia atrás
    
    abduccion: True si el brazo está abducido (separado lateralmente).
    hombro_elevado: True si el hombro está encogido/elevado.
    apoyo_brazo: True si el brazo tiene apoyo o gravedad asistida.
    """
    ang = abs(angulo_grados)

    if ang < 20:
        score = 1
    elif ang < 45:
        score = 2
    elif ang < 90:
        score = 3
    else:
        score = 4

    # Extensión hacia atrás > 20° también puntúa al menos 2
    if angulo_grados < -20:
        score = max(score, 2)

    if abduccion or hombro_elevado:
        score += 1
    if apoyo_brazo:
        score -= 1

    return max(1, score)  # Mínimo 1


def puntuacion_antebrazo(angulo_grados):
    """
    Puntuación del antebrazo según REBA.
    
    angulo_grados: ángulo en el codo (entre brazo y antebrazo).
      - 60-100° = posición neutra (score 1)
      - Fuera de ese rango = score 2
    """
    ang = abs(angulo_grados)

    if 60 <= ang <= 100:
        return 1
    else:
        return 2


def puntuacion_muneca(angulo_grados, torsion=False):
    """
    Puntuación de la muñeca según REBA.
    
    angulo_grados: desviación de la muñeca respecto al antebrazo.
      - 0-15°  = posición neutra (score 1)
      - >15°   = desviación excesiva (score 2)
    
    torsion: True si hay torsión/desviación lateral de la muñeca.
    """
    ang = abs(angulo_grados)

    if ang <= 15:
        score = 1
    else:
        score = 2

    if torsion:
        score += 1

    return score


# ==============================================================================
# TABLAS REBA (Tablas A y B oficiales)
# ==============================================================================

# Tabla A: [tronco][cuello][piernas] → score_A base
# Índices: tronco 1-5, cuello 1-3, piernas 1-4
# Dimensiones: tabla_A[tronco-1][cuello-1][piernas-1]
TABLA_A = [
    # tronco=1
    [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6]],
    # tronco=2
    [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
    # tronco=3
    [[2, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    # tronco=4
    [[3, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    # tronco=5
    [[4, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
]

# Tabla B: [brazo][antebrazo][muñeca] → score_B base
# Índices: brazo 1-6, antebrazo 1-2, muñeca 1-3
TABLA_B = [
    # brazo=1
    [[1, 2, 2], [1, 2, 3]],
    # brazo=2
    [[1, 2, 3], [2, 3, 4]],
    # brazo=3
    [[3, 4, 5], [4, 5, 5]],
    # brazo=4
    [[4, 5, 5], [5, 6, 7]],
    # brazo=5
    [[6, 7, 8], [7, 8, 8]],
    # brazo=6
    [[7, 8, 8], [8, 9, 9]],
]

# Tabla C: [score_A_ajustado][score_B_ajustado] → score_C
# Índices: A 1-12, B 1-12
TABLA_C = [
    #      B: 1   2   3   4   5   6   7   8   9  10  11  12
    [1,  1,  1,  2,  3,  3,  4,  5,  6,  7,  7,  7],   # A=1
    [1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  7,  8],   # A=2
    [2,  3,  3,  3,  4,  5,  6,  7,  7,  8,  8,  8],   # A=3
    [3,  4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9],   # A=4
    [4,  4,  4,  5,  6,  7,  8,  8,  9,  9, 10, 10],   # A=5
    [6,  6,  6,  7,  8,  8,  9,  9, 10, 10, 11, 11],   # A=6
    [7,  7,  7,  8,  9,  9,  9, 10, 11, 11, 11, 12],   # A=7
    [8,  8,  8,  9, 10, 10, 10, 11, 11, 12, 12, 12],   # A=8
    [9,  9,  9,  9, 10, 10, 11, 11, 12, 12, 12, 12],   # A=9
    [10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 12],  # A=10
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],  # A=11
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],  # A=12
]


def lookup_tabla_A(tronco, cuello, piernas):
    """Busca en Tabla A. Índices 1-based."""
    t = max(1, min(int(tronco), 5)) - 1
    c = max(1, min(int(cuello), 3)) - 1
    p = max(1, min(int(piernas), 4)) - 1
    return TABLA_A[t][c][p]


def lookup_tabla_B(brazo, antebrazo, muneca):
    """Busca en Tabla B. Índices 1-based."""
    br = max(1, min(int(brazo), 6)) - 1
    an = max(1, min(int(antebrazo), 2)) - 1
    mu = max(1, min(int(muneca), 3)) - 1
    return TABLA_B[br][an][mu]


def lookup_tabla_C(score_a, score_b):
    """Busca en Tabla C. Índices 1-based."""
    a = max(1, min(int(score_a), 12)) - 1
    b = max(1, min(int(score_b), 12)) - 1
    return TABLA_C[a][b]


# ==============================================================================
# PUNTUACIÓN REBA COMPLETA
# ==============================================================================

def calcular_reba_completo(
    angulo_tronco,
    angulo_cuello,
    angulo_rodilla,
    angulo_brazo,
    angulo_antebrazo_val,
    angulo_muneca_val,
    # Modificadores opcionales
    torsion_tronco=False,
    inclinacion_lateral_tronco=False,
    torsion_cuello=False,
    inclinacion_lateral_cuello=False,
    soporte_bilateral=True,
    abduccion_brazo=False,
    hombro_elevado=False,
    apoyo_brazo=False,
    torsion_muneca=False,
    # Cargas y actividad
    carga_kg=0,
    tipo_agarre="bueno",
    actividad_repetitiva=False,
    postura_estatica=False,
    cambios_rapidos=False,
):
    """
    Calcula el score REBA completo.
    
    Parámetros de ángulos (todos en grados):
        angulo_tronco: flexión del tronco respecto a vertical (0=erguido)
        angulo_cuello: flexión del cuello respecto al tronco (0=neutro)
        angulo_rodilla: flexión de rodilla (0=extendida)
        angulo_brazo: elevación del brazo respecto al tronco (0=pegado)
        angulo_antebrazo_val: ángulo en el codo brazo-antebrazo
        angulo_muneca_val: desviación de la muñeca
    
    Retorna dict con todos los scores intermedios y el nivel de riesgo.
    """

    # --- GRUPO A ---
    s_tronco = puntuacion_tronco(angulo_tronco, torsion_tronco, inclinacion_lateral_tronco)
    s_cuello = puntuacion_cuello(angulo_cuello, torsion_cuello, inclinacion_lateral_cuello)
    s_piernas = puntuacion_piernas(angulo_rodilla, soporte_bilateral)

    score_A_tabla = lookup_tabla_A(s_tronco, s_cuello, s_piernas)

    # Modificador de carga/fuerza
    if carga_kg < 5:
        mod_carga = 0
    elif carga_kg < 10:
        mod_carga = 1
    else:
        mod_carga = 2
    # +1 si hay fuerza brusca (no implementado en GUI por simplicidad)

    score_A_final = score_A_tabla + mod_carga

    # --- GRUPO B ---
    s_brazo = puntuacion_brazo(angulo_brazo, abduccion_brazo, hombro_elevado, apoyo_brazo)
    s_antebrazo = puntuacion_antebrazo(angulo_antebrazo_val)
    s_muneca = puntuacion_muneca(angulo_muneca_val, torsion_muneca)

    score_B_tabla = lookup_tabla_B(s_brazo, s_antebrazo, s_muneca)

    # Modificador de agarre
    agarre_map = {"bueno": 0, "regular": 1, "malo": 2, "inaceptable": 3}
    mod_agarre = agarre_map.get(tipo_agarre, 0)

    score_B_final = score_B_tabla + mod_agarre

    # --- TABLA C ---
    score_C = lookup_tabla_C(score_A_final, score_B_final)

    # Modificador de actividad
    mod_actividad = 0
    if actividad_repetitiva:
        mod_actividad += 1
    if postura_estatica:
        mod_actividad += 1
    if cambios_rapidos:
        mod_actividad += 1

    score_reba = score_C + mod_actividad
    score_reba = max(1, min(score_reba, 15))

    # Nivel de riesgo y acción
    if score_reba == 1:
        nivel = "Insignificante"
        accion = "Sin acción necesaria"
        color = (0, 255, 0)
    elif score_reba <= 3:
        nivel = "Bajo"
        accion = "Puede ser necesaria acción"
        color = (0, 200, 100)
    elif score_reba <= 7:
        nivel = "Medio"
        accion = "Acción necesaria"
        color = (0, 200, 255)
    elif score_reba <= 10:
        nivel = "Alto"
        accion = "Acción necesaria pronto"
        color = (0, 100, 255)
    elif score_reba <= 15:
        nivel = "Muy alto"
        accion = "Acción necesaria de inmediato"
        color = (0, 0, 255)
    else:
        nivel = "Muy alto"
        accion = "Acción necesaria de inmediato"
        color = (0, 0, 255)

    return {
        # Scores individuales
        "s_tronco": s_tronco,
        "s_cuello": s_cuello,
        "s_piernas": s_piernas,
        "s_brazo": s_brazo,
        "s_antebrazo": s_antebrazo,
        "s_muneca": s_muneca,
        # Scores de tabla
        "score_A_tabla": score_A_tabla,
        "score_B_tabla": score_B_tabla,
        # Modificadores
        "mod_carga": mod_carga,
        "mod_agarre": mod_agarre,
        "mod_actividad": mod_actividad,
        # Scores ajustados
        "score_A": score_A_final,
        "score_B": score_B_final,
        "score_C": score_C,
        # Score final
        "score_reba": score_reba,
        "nivel_riesgo": nivel,
        "accion": accion,
        "color_riesgo": color,
    }


# ==============================================================================
# COMPATIBILIDAD CON CÓDIGO ANTERIOR (mantiene las funciones originales)
# ==============================================================================

def reba_total(score_A, score_B):
    """Compatibilidad. Usa calcular_reba_completo para resultados precisos."""
    score_C = lookup_tabla_C(score_A, score_B)
    if score_C <= 1:
        riesgo = "Insignificante"
    elif score_C <= 3:
        riesgo = "Bajo"
    elif score_C <= 7:
        riesgo = "Medio"
    elif score_C <= 10:
        riesgo = "Alto"
    else:
        riesgo = "Muy alto"
    return score_C, riesgo

"""
Estructura REBA:
  Grupo A: Tronco + Cuello + Piernas → Tabla A → Score A → +Carga → Score A ajustado
  Grupo B: Brazo + Antebrazo + Muñeca → Tabla B → Score B → +Agarre → Score B ajustado
  Tabla C (Score A ajustado, Score B ajustado) → Score C → +Actividad → Score REBA Final
"""