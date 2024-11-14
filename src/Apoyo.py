import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import *
import yfinance as yf
import matplotlib.pyplot as plt

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

def calcular_renta_vitalicia(FV, tasa_interes_mensual, n_meses_vida):
    return FV * tasa_interes_mensual / (1 - (1 + tasa_interes_mensual) ** (-n_meses_vida))

def generar_tabla_ahorro_mensual(salario, tasa_aporte, tasa_interes_mensual, meses):
    aporte_mensual = salario * tasa_aporte  # Aporte mensual
    saldo_acumulado = 0
    
    tabla_ahorro = []

    for mes in range(1, meses + 1):
        # Calculamos el saldo acumulado con interés compuesto
        saldo_acumulado = (saldo_acumulado + aporte_mensual) * (1 + tasa_interes_mensual)
        
        # Añadimos el resultado a la tabla
        tabla_ahorro.append({
            "Mes": mes, 
            "Aporte Mensual": round(aporte_mensual, 2),  # Redondeamos el aporte mensual a dos decimales
            "Saldo Acumulado": round(saldo_acumulado, 2)  # Redondeamos el saldo acumulado
        })

    # Convertimos la lista de resultados en un DataFrame de pandas
    return pd.DataFrame(tabla_ahorro)
    
def calcular_reserva_actuarial(salario, tasa_aporte, tasa_interes_mensual, meses_aporte):
    saldo_acumulado = 0
    for mes in range(1, meses_aporte + 1):
        saldo_acumulado = (saldo_acumulado + salario * tasa_aporte) * (1 + tasa_interes_mensual)
    return round(saldo_acumulado, 2)
# Función para generar la tabla de ahorro mes a mes
def generar_tabla_ahorro_mensual(salario, tasa_aporte, tasa_interes_mensual, meses):
    aporte_mensual = salario * tasa_aporte  # Aporte mensual
    saldo_acumulado = 0
    
    tabla_ahorro = []

    for mes in range(1, meses + 1):
        # Calculamos el saldo acumulado con interés compuesto
        saldo_acumulado = (saldo_acumulado + aporte_mensual) * (1 + tasa_interes_mensual)
        
        # Añadimos el resultado a la tabla
        tabla_ahorro.append({
            "Mes": mes, 
            "Aporte Mensual": round(aporte_mensual, 2),  # Redondeamos el aporte mensual a dos decimales
            "Saldo Acumulado": round(saldo_acumulado, 2)  # Redondeamos el saldo acumulado
        })

    # Convertimos la lista de resultados en un DataFrame de pandas
    return pd.DataFrame(tabla_ahorro)


def TMensual(Tmort):
    """
    Interpola los valores de l(x) en pasos de 1/12 (mensual) y calcula q(x) y p(x).
    
    Parameters:
    Tmort (DataFrame): DataFrame que contiene las columnas 'x' y 'l(x)'.
    
    Returns:
    DataFrame: DataFrame con valores interpolados de x, l(x), q(x) y p(x).
    """
    
    # Aseguramos que Tmort tiene las columnas adecuadas
    if 'x' not in Tmort.columns or 'l(x)' not in Tmort.columns:
        raise ValueError("El DataFrame debe contener las columnas 'x' y 'l(x)'")
    
    # Obtener los valores de x y l(x)
    x_values = Tmort['x'].values
    lx_values = Tmort['l(x)'].values
    
    # Crear una función de interpolación
    interp_function = interp1d(x_values, lx_values, kind='linear', fill_value='extrapolate')
    
    # Crear los nuevos valores de x con un paso de 1/12, hasta 110 (incluyendo 110)
    new_x = np.arange(x_values[0], 110 + 1/12, 1/12)  # Incrementos mensuales hasta 110
    
    # Calcular los valores de l(x) interpolados
    new_lx = interp_function(new_x)
    
    # Inicializar los arrays para q(x) y p(x)
    qx_values = np.zeros(len(new_x) - 1)  # Inicializar el array para q(x)
    px_values = np.zeros(len(new_x) - 1)  # Inicializar el array para p(x)

    # Calcular q(x) y p(x) para cada nuevo valor de x
    for i in range(len(new_x)-1):
        if new_lx[i] > 0:  # Evitar división por cero
            qx_values[i] = 1 - (new_lx[i + 1] / new_lx[i])  # q(x) = 1 - l(x + 1/12) / l(x)
        else:
            qx_values[i] = 0  # Si l(x) es 0, q(x) también debe ser 0

        px_values[i] = 1 - qx_values[i]  # p(x) = 1 - q(x)


    # Crear un nuevo DataFrame con los valores interpolados y calculados
    interpolated_df = pd.DataFrame({
        'x': new_x,  # Incluir hasta 110
        'l(x)': new_lx,  # Excluir el último por la misma razón
        'q(x)': np.append(qx_values, 1),  # Añadir el valor de q(110)
        'p(x)': np.append(px_values, 0)  # Añadir el valor de p(110)
    })
    
    return interpolated_df

def tmort_filtered(df,edad):
    qx_vec=df[df['x']>=edad]
    return qx_vec

def datos_sesgados(sesgo, cantidad, maximo):
    #distribucion normal sesgada
    aleatorios = skewnorm.rvs(sesgo, loc = 2.5, size = cantidad, random_state = 0)
    aleatorios = aleatorios + abs(aleatorios.min())
    aleatorios = np.round(aleatorios, 2)/aleatorios.max() * maximo
    return np.round(aleatorios, 1)
    

def extrae(ticker, start_date, end_date):
    """
    Descarga los precios históricos de un ticker y grafica su precio de cierre.

    Args:
    ticker (str): El símbolo del ticker (por ejemplo, 'AAPL' para Apple, 'TSLA' para Tesla, 'BTC-CAD' para Bitcoin en CAD).
    start_date (str): La fecha de inicio en formato 'YYYY-MM-DD'.
    end_date (str): La fecha de fin en formato 'YYYY-MM-DD'.
    """
    # 1. Descargar los precios históricos usando yfinance
    data = yf.download(ticker, start=start_date, end=end_date)  # Especificamos el rango de fechas
    #2. Verificar si los datos se descargaron correctamente
    if data.empty:
        print(f"No se encontraron datos para el ticker {ticker} en el rango de fechas especificado.")
        return
    #3. Aplanar el MultiIndex de las columnas
    data.columns = [col[0] for col in data.columns]
    #4. plot
    data['Adj Close'].plot(title=f"Precio de Cierre de {ticker}", figsize=(10, 6))
    plt.xlabel("Fecha")
    plt.ylabel("Precio de Cierre")
    plt.grid(True)
    plt.show()
    return data
# Funcion que realiza graficos de lineas
def lineas_multiples(data, title='Gráfico de Múltiples Líneas', xlabel='Eje X', ylabel='Eje Y'):
    """
    Función para graficar múltiples líneas con colores aleatorios.
    
    Parámetros:
    - data: DataFrame de pandas con los datos a graficar.
    - title: Título del gráfico.
    - xlabel: Etiqueta del eje X.
    - ylabel: Etiqueta del eje Y.
    """
    plt.figure(figsize=(10, 6))  # Tamaño de la figura
    num_lines = data.shape[1]  # Número de líneas (columnas en el DataFrame)
    
    # Generar un color aleatorio para cada línea
    colors = np.random.rand(num_lines, 3)  # Colores aleatorios en RGB

    for i in range(num_lines):
        plt.plot(np.arange(1, data.shape[0] + 1), data.iloc[:, i], color=colors[i], label=f'Línea {i + 1}')  # Acceso corregido
    
    # Configurar título y etiquetas
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Mostrar la cuadrícula
    plt.grid(True)
    
    # Mostrar leyenda
    #plt.legend()
    
    # Mostrar el gráfico
    plt.show()
    
#Procedemos a crear una funcion que modele el proceso estocastico
def MGB_acciones(data,col_name,steps):

    #Los parametros de la funcion son:
    #data -> pandas df que contiene los retornos logaritmicos del activo en dias
    #col_name -> nombre de la columna de los retornos
    #steps -> numero de pasos hacia adelante en el proceso estocastico
    data['retornos']=(data['Adj Close'] / data['Adj Close'].shift(1)).apply(lambda x: np.log(x))
    mu=data['retornos'].mean()
    sigma = data['retornos'].std()
    dt=1/len(data) # cantidad dias en muestra--1/152
    s0=data[col_name].iloc[-1] #Tomamos el ultimo elemento de los precios, ya que este sera el punto de partida
    et=np.random.normal(loc=0, scale=1, size=steps) # Vector de numeros aletorios de media 0 y desviacion estandar 1
    euler = np.exp(((mu - (sigma*sigma) / 2) * dt) + sigma * et * np.sqrt(dt))

    st_1=[]

    for i in range(0,steps):

        if i==0:
            s=s0*euler[0]
        else:
            s=st_1[i-1]*euler[i]

        st_1.append(s)
            

    return st_1

   

    
