#!pip install yfinance
#!pip install --upgrade pyarrow
#!pip install lightgbm
#!pip install statsmodels
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import *
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

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
def filtrar_columnas_por_categoria_y_correlacion(df_muestra, dict_path, categorias_buscar, var_resp, peso):
    # Inicializar lista para almacenar resultados
    resultados = []
    
    # Iterar sobre las categorías que deseas buscar
    for categoria_buscar in categorias_buscar:
        # Filtrar dict_path por la categoría deseada
        categorias_filtradas = dict_path[dict_path['CATEGORÍA'] == categoria_buscar]
        # Obtener los campos asociados a esa categoría
        campos_de_categoria = categorias_filtradas['Factores'].tolist()
        # Filtrar df_muestra para solo incluir las columnas válidas
        columnas_validas = [col for col in campos_de_categoria if col in df_muestra.columns]
        df_filtrado = df_muestra[columnas_validas]
        print(f"DataFrame filtrado por la categoría '{categoria_buscar}':")
        # Verificar que var_resp y peso están en las columnas
        if var_resp not in df_muestra.columns or peso not in df_muestra.columns:
            print(f"Error: Las columnas '{var_resp}' o '{peso}' no están presentes en df_muestra.")
            return None
        # Agregar la columna 'resp' al DataFrame filtrado
        df_filtrado['resp'] = df_muestra['resp']
        # Filtrar las columnas numéricas
        df_filtrado_numerico = df_filtrado.select_dtypes(include='number')
        # Calcular la matriz de correlación
        corr_matrix = df_filtrado_numerico.corr()
        # Filtrar las correlaciones con la variable 'resp'
        corr_resp = corr_matrix['resp'].abs().reset_index()
        corr_resp.columns = ['Factor', 'Correlación']
        # Eliminar la correlación de la variable consigo misma
        corr_resp = corr_resp[corr_resp['Factor'] != 'resp']
        # corr_resp = corr_resp[corr_resp['Correlación'] > 0.051]
        # Ordenar por la magnitud de la correlación de mayor a menor
        corr_resp = corr_resp.sort_values(by='Correlación', ascending=False)
        # Seleccionar las 50 correlaciones más fuertes
        top_5_corr = corr_resp.head(5)
        print(f"Top 5 correlaciones más fuertes para la categoría '{categoria_buscar}':")
        print(top_5_corr)

        # Crear una matriz de correlación para los pares seleccionados
        top_corr_matrix = corr_matrix.loc[top_5_corr['Factor']]        
        # Dibujar el diagrama de correlación
        #plt.figure(figsize=(12, 8))
        #sns.heatmap(top_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        #plt.title(f"Top 5 Correlaciones para la categoría '{categoria_buscar}' y sus factores")
        #plt.show()
        # Almacenar los resultados para cada categoría
        resultados.append(top_5_corr)
    
    # Si deseas consolidar los resultados en un solo DataFrame, puedes hacerlo aquí
    if resultados:
        df_resultado_final = pd.concat(resultados, keys=categorias_buscar)
        return df_resultado_final
    return resultados
def graficar_barras_linea(df, x_col, y_col_barras, y_col_linea, titulo="Gráfico combinado de barras y línea"):
    """
    Función para graficar un diagrama de barras con una línea superpuesta.
    
    Parámetros:
    - df: DataFrame de Pandas.
    - x_col: Nombre de la columna para el eje x (categorías).
    - y_col_barras: Nombre de la columna para el eje y de las barras (pesos).
    - y_col_linea: Nombre de la columna para el eje y de la línea (variable de respuesta).
    - titulo: Título del gráfico (opcional).
    """

    # Crear el gráfico
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Graficar las barras
    ax1.bar(df[x_col], df[y_col_barras], color='skyblue', label=y_col_barras, alpha=0.7)

    # Etiquetas para el eje de las barras
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col_barras, color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    # Crear un segundo eje Y para la línea
    ax2 = ax1.twinx()  # Comparte el mismo eje X

    # Graficar la línea
    ax2.plot(df[x_col], df[y_col_linea], color='orange', marker='o', label=y_col_linea)

    # Etiquetas para el eje de la línea
    ax2.set_ylabel(y_col_linea, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Título
    plt.title(titulo)

    # Rotar etiquetas del eje X si es necesario
    plt.xticks(rotation=90)

    # Asegurar que no se solapen los gráficos
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()
def generar_formato_exposicion_homogenea(df, columna_interes, columna_exposicion, num_beans):
    """
    Genera un formato de beans con exposición homogénea basados en la exposición de otra columna y devuelve la columna
    de interés con el formato aplicado.

    :param df: DataFrame que contiene los datos.
    :param columna_interes: Nombre de la columna de interés que se quiere agrupar en bins.
    :param columna_exposicion: Nombre de la columna que contiene la exposición o peso.
    :param distribucion: Tipo de distribución ('normal', 'uniforme', etc.) para ajustar los beans (parámetro no usado en esta versión).
    :param num_beans: Número de beans o grupos a crear.
    :return: Serie con la columna de interés formateada en beans de exposición constante.
    """
    
    # Ordenar el DataFrame por la columna de interés para crear beans adecuados
    df = df.sort_values(by=columna_interes).reset_index(drop=True)

    # Calcular la exposición total y la exposición por cada bean
    exposicion_total = df[columna_exposicion].sum()
    exposicion_por_bean = exposicion_total / num_beans
    
    # Variable para rastrear la exposición acumulada y asignar beans
    exposicion_acumulada = 0
    limites_beans = []
    current_bean = 1
    limite_actual = df[columna_interes].iloc[0]  # Inicia en el valor mínimo
    
    # Recorrer el DataFrame fila por fila para crear los bins basados en exposición
    for i, fila in df.iterrows():
        exposicion_acumulada += fila[columna_exposicion]
        
        # Si alcanzamos la exposición deseada para este bin, guardar el límite superior
        if exposicion_acumulada >= exposicion_por_bean:
            limite_actual = fila[columna_interes]
            limites_beans.append(limite_actual)
            exposicion_acumulada = 0
            current_bean += 1
            if current_bean > num_beans:
                break  # No más beans después del número establecido
    
    # Si el número de beans es menor al solicitado, añadimos el máximo valor como último límite
    if len(limites_beans) < num_beans:
        limites_beans.append(df[columna_interes].max())
    
    # Usar pd.cut para aplicar los límites de los beans
    df['Columna_formateada'] = pd.cut(df[columna_interes], bins=[-np.inf] + limites_beans, duplicates='drop')
    
    return df['Columna_formateada']
def calculate_bic(model):
    n = len(model.model.endog)  # Número de observaciones
    k = model.df_model + 1  # Número de parámetros (df_model incluye los coeficientes, sumamos 1 por el intercepto)
    bic = np.log(n) * k - 2 * model.llf  # BIC = ln(n) * k - 2 * log-verosimilitud
    return bic
