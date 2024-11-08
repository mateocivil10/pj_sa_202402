def valor_presente(FV, r, n):
    """
    Calcula el valor presente (VP) a partir de un valor futuro (FV),
    una tasa de interés (r) y el número de periodos (n).
    
    :param FV: Valor futuro (Future Value)
    :param r: Tasa de interés por periodo (tasa de descuento)
    :param n: Número de periodos (años, meses, etc.)
    :return: Valor presente (PV)
    """
    VP = FV / (1 + r) ** n
    return VP

def valor_futuro(PV, r, n):
    """
    Calcula el valor futuro (VF) a partir de un valor presente (PV),
    una tasa de interés (r) y el número de periodos (n).
    
    :param PV: Valor presente (Present Value)
    :param r: Tasa de interés por periodo
    :param n: Número de periodos
    :return: Valor futuro (FV)
    """
    VF = PV * (1 + r) ** n
    return VF

def tasa_efectiva_anual(tasa_nominal, m):
    """
    Convierte una tasa nominal anual a su tasa efectiva anual equivalente.
    :param tasa_nominal: Tasa nominal anual.
    :param m: Número de periodos de capitalización al año (12 para mensual, 4 para trimestral).
    :return: Tasa efectiva anual.
    """
    tasa_efectiva = (1 + tasa_nominal / m) ** m - 1
    return tasa_efectiva


def tasa_efectiva_anual_desde_semestre(tasa_efectiva_semestral):
    """
    Calcula la tasa efectiva anual a partir de una tasa efectiva semestral.
    :param tasa_efectiva_semestral: Tasa efectiva semestral.
    :return: Tasa efectiva anual.
    """
    tasa_efectiva_anual = (1 + tasa_efectiva_semestral) ** 2 - 1
    return tasa_efectiva_anual

def tasa_semestral_desde_anual(tasa_efectiva_anual):
    """
    Calcula la tasa semestral a partir de una tasa efectiva anual.
    :param tasa_efectiva_anual: Tasa efectiva anual.
    :return: Tasa semestral.
    """
    tasa_semestral = (1 + tasa_efectiva_anual) ** 0.5 - 1
    return tasa_semestral
def tasa_nominal_mensual(i_e):
    """
    Calcula la tasa nominal mensual a partir de una tasa efectiva anual.
    :param i_e: Tasa efectiva anual.
    :return: Tasa nominal mensual.
    """
    m = 12  # Capitalización mensual
    i_n = m * ((1 + i_e) ** (1 / m) - 1)
    return i_n


    
