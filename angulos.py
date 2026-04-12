import numpy as np


def calcular_angulo_3puntos(a, b, c):
    """
    Calcula el ángulo en el punto B formado por los segmentos BA y BC.
    Ángulos en grados [0, 180].
    """
    a = np.array(a[:2])  # Solo x, y
    b = np.array(b[:2])
    c = np.array(c[:2])

    ba = a - b
    bc = c - b

    norma_ba = np.linalg.norm(ba)
    norma_bc = np.linalg.norm(bc)

    if norma_ba == 0 or norma_bc == 0:
        return 0.0

    cos_angulo = np.dot(ba, bc) / (norma_ba * norma_bc)
    cos_angulo = np.clip(cos_angulo, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angulo))


def angulo_con_vertical(punto_superior, punto_inferior):
    """
    Calcula el ángulo de un segmento corporal respecto a la VERTICAL.
    En imagen, la vertical es el eje Y (hacia abajo).
    - 0°  = perfectamente vertical (erguido)
    - 90° = horizontal
    """
    p_sup = np.array(punto_superior[:2])
    p_inf = np.array(punto_inferior[:2])

    segmento = p_inf - p_sup  # Vector desde superior a inferior

    vertical = np.array([0, 1])  # Vertical en coordenadas imagen (Y hacia abajo)

    norma = np.linalg.norm(segmento)
    if norma == 0:
        return 0.0

    cos_angulo = np.dot(segmento, vertical) / norma
    cos_angulo = np.clip(cos_angulo, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angulo))


def angulo_con_horizontal(punto_a, punto_b):
    """
    Calcula el ángulo de un segmento respecto a la HORIZONTAL.
    Útil para muñeca o antebrazo respecto al suelo.
    """
    pa = np.array(punto_a[:2])
    pb = np.array(punto_b[:2])

    segmento = pb - pa
    horizontal = np.array([1, 0])

    norma = np.linalg.norm(segmento)
    if norma == 0:
        return 0.0

    cos_angulo = abs(np.dot(segmento, horizontal)) / norma
    cos_angulo = np.clip(cos_angulo, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angulo))


def angulo_tronco_vertical(hombro, cadera):
    """
    Ángulo del tronco respecto a la vertical.
    Segmento: cadera -> hombro (de abajo hacia arriba).
    0° = erguido, 90° = tumbado.
    """
    # En coordenadas imagen Y está invertido: hombro tiene menor Y que cadera
    # Usamos cadera como base e invertimos vertical para que 0° sea erguido
    p_hombro = np.array(hombro[:2])
    p_cadera = np.array(cadera[:2])

    segmento = p_hombro - p_cadera  # Vector de cadera a hombro (hacia arriba en imagen = Y negativo)
    vertical_up = np.array([0, -1])  # Vertical hacia arriba en coordenadas imagen

    norma = np.linalg.norm(segmento)
    if norma == 0:
        return 0.0

    cos_angulo = np.dot(segmento, vertical_up) / norma
    cos_angulo = np.clip(cos_angulo, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angulo))


def angulo_cuello(oreja, hombro, cadera):
    """
    Ángulo del cuello respecto al tronco.
    Compara la dirección cabeza-hombro con hombro-cadera.
    0° = cabeza alineada con tronco, valores mayores = flexión.
    """
    p_oreja = np.array(oreja[:2])
    p_hombro = np.array(hombro[:2])
    p_cadera = np.array(cadera[:2])

    vec_cabeza = p_oreja - p_hombro    # Dirección del cuello
    vec_tronco = p_cadera - p_hombro   # Dirección del tronco

    norma_cab = np.linalg.norm(vec_cabeza)
    norma_tronco = np.linalg.norm(vec_tronco)

    if norma_cab == 0 or norma_tronco == 0:
        return 0.0

    cos_angulo = np.dot(vec_cabeza, vec_tronco) / (norma_cab * norma_tronco)
    cos_angulo = np.clip(cos_angulo, -1.0, 1.0)
    # El ángulo entre oreja-hombro y cadera-hombro: si están alineados = 180°
    # Convertimos a desviación respecto al tronco
    angulo_raw = np.degrees(np.arccos(cos_angulo))
    return abs(180 - angulo_raw)  # 0° = alineado con tronco


def angulo_brazo_tronco(hombro, codo, cadera):
    """
    Ángulo del brazo (húmero) respecto al tronco.
    Según REBA: ángulo entre el segmento hombro-codo y la línea del tronco.
    0° = brazo pegado al cuerpo, 90° = brazo horizontal.
    """
    p_hombro = np.array(hombro[:2])
    p_codo = np.array(codo[:2])
    p_cadera = np.array(cadera[:2])

    vec_brazo = p_codo - p_hombro
    vec_tronco = p_cadera - p_hombro

    norma_brazo = np.linalg.norm(vec_brazo)
    norma_tronco = np.linalg.norm(vec_tronco)

    if norma_brazo == 0 or norma_tronco == 0:
        return 0.0

    cos_angulo = np.dot(vec_brazo, vec_tronco) / (norma_brazo * norma_tronco)
    cos_angulo = np.clip(cos_angulo, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angulo))


def angulo_antebrazo(hombro, codo, muneca):
    """
    Ángulo del antebrazo en el codo.
    Ángulo entre el brazo (hombro-codo) y el antebrazo (codo-muñeca).
    Según REBA: se evalúa si está entre 60-100° (posición neutra) o fuera.
    """
    return calcular_angulo_3puntos(hombro, codo, muneca)


def angulo_muneca(codo, muneca, indice):
    """
    Desviación de la muñeca respecto al antebrazo.
    Ángulo entre el antebrazo (codo-muñeca) y la mano (muñeca-índice).
    """
    return calcular_angulo_3puntos(codo, muneca, indice)

'''
def angulo_pierna(cadera, rodilla, tobillo):
    """
    Ángulo de la pierna en la rodilla.
    Útil para REBA grupo A (piernas).
    """
    return calcular_angulo_3puntos(cadera, rodilla, tobillo)
'''

def calcular_angulo(a, b, c):
    """
    Compatibilidad hacia atrás. Usa calcular_angulo_3puntos.
    """
    return calcular_angulo_3puntos(a, b, c)