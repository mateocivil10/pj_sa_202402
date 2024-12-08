import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def obtener_correlaciones_altas(df, umbral=0.8, excluir_columnas=None):
    """
    Obtiene una tabla de correlaciones absolutas mayores a un umbral dado.

    Parámetros:
    - df: DataFrame de pandas que contiene los datos.
    - umbral: Valor absoluto mínimo de correlación para incluir en los resultados (por defecto 0.8).
    - excluir_columnas: Lista de columnas a excluir antes de calcular la correlación (opcional).

    Retorna:
    - DataFrame con las correlaciones altas, incluyendo las columnas 'var1', 'var2' y 'correlacion'.
    """
    # Excluir columnas si se especifican
    if excluir_columnas:
        df = df.drop(columns=excluir_columnas)
    
    # Calcular matriz de correlación
    correlacion = df.corr()

    # Filtrar correlaciones mayores al umbral
    correlaciones_mayores = (
        correlacion.where((abs(correlacion) > umbral) & (correlacion < 1))
        .stack()  # Convertir matriz a formato largo
        .reset_index()  # Reiniciar índice para obtener DataFrame
        .rename(columns={'level_0': 'var1', 'level_1': 'var2', 0: 'correlacion'})  # Renombrar columnas
    )

    # Eliminar duplicados (ej. x1-x2 y x2-x1 son lo mismo)
    correlaciones_mayores = correlaciones_mayores[correlaciones_mayores['var1'] < correlaciones_mayores['var2']]
    
    return correlaciones_mayores


def calcular_vif(df, excluir_columnas=None):
    """
    Calcula el Factor de Inflación de la Varianza (VIF) para cada variable de un DataFrame.

    Parámetros:
    - df: DataFrame de pandas que contiene las variables explicativas.
    - excluir_columnas: Lista de columnas a excluir antes del cálculo (opcional).

    Retorna:
    - DataFrame con columnas 'Variable' y 'VIF'.
    """
    # Excluir columnas si se especifican
    if excluir_columnas:
        df = df.drop(columns=excluir_columnas)

    # Agregar una constante para el cálculo de VIF
    df_const = sm.add_constant(df)

    # Calcular el VIF para cada variable
    vif_data = pd.DataFrame()
    vif_data["Variable"] = df_const.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df_const.values, i)
        for i in range(df_const.shape[1])
    ]

    return vif_data


def graficar_histograma(df, columna, bins=10, titulo="Histograma", xlabel="Valores", ylabel="Frecuencia"):
    """
    Grafica un histograma de una variable específica de un DataFrame.
    
    Parámetros:
    - df: DataFrame de pandas que contiene los datos.
    - columna: Nombre de la columna que se desea graficar (str).
    - bins: Número de bins en el histograma (opcional, por defecto 10).
    - titulo: Título del gráfico (opcional).
    - xlabel: Etiqueta para el eje X (opcional).
    - ylabel: Etiqueta para el eje Y (opcional).
    """
    # Comprobamos si la columna existe en el DataFrame
    if columna not in df.columns:
        print(f"La columna '{columna}' no se encuentra en el DataFrame.")
        return
    
    # Graficar el histograma
    plt.figure(figsize=(8, 6))
    plt.hist(df[columna].dropna(), bins=bins, color="skyblue", edgecolor="black")
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
    

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


def graficar_dispersion(df, x_columna, y_columna, titulo="Gráfico de Dispersión", xlabel=None, ylabel=None):
    """
    Grafica un gráfico de dispersión entre dos variables de un DataFrame.

    Parámetros:
    - df: DataFrame de pandas que contiene los datos.
    - x_columna: Nombre de la columna para el eje X (str).
    - y_columna: Nombre de la columna para el eje Y (str).
    - titulo: Título del gráfico (opcional).
    - xlabel: Etiqueta para el eje X (opcional, usa el nombre de la columna por defecto).
    - ylabel: Etiqueta para el eje Y (opcional, usa el nombre de la columna por defecto).
    """
    # Comprobamos si las columnas existen en el DataFrame
    if x_columna not in df.columns or y_columna not in df.columns:
        print(f"Una o ambas columnas ('{x_columna}', '{y_columna}') no se encuentran en el DataFrame.")
        return
    
    # Graficar el gráfico de dispersión
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_columna, y=y_columna)
    plt.title(titulo)
    plt.xlabel(xlabel if xlabel else x_columna)
    plt.ylabel(ylabel if ylabel else y_columna)
    plt.show()
    
def calculate_bic(model):
    n = len(model.model.endog)  # Número de observaciones
    k = model.df_model + 1  # Número de parámetros (df_model incluye los coeficientes, sumamos 1 por el intercepto)
    bic = np.log(n) * k - 2 * model.llf  # BIC = ln(n) * k - 2 * log-verosimilitud
    return bic

def graficar_coeficientes(modelo_resultado):
    """
    Genera un gráfico de barras horizontales de los coeficientes del modelo GLM.
    
    Parámetros:
    modelo_resultado: objeto de resultado del modelo GLM ajustado.
    """
    # Extraer los coeficientes y los nombres de las variables
    coeficientes = modelo_resultado.params
    coeficientes = coeficientes.drop('const')  # Eliminar la constante
    coeficientes = coeficientes.sort_values(ascending=False)
    
    # Crear un DataFrame para la visualización
    df_coef = pd.DataFrame({'Variable': coeficientes.index, 'Coeficiente': coeficientes.values})
    
    # Configuración del gráfico
    plt.figure(figsize=(10, 6))
    colors = ['green' if coef > 0 else 'red' for coef in df_coef['Coeficiente']]
    plt.barh(df_coef['Variable'], df_coef['Coeficiente'], color=colors)
    plt.xlabel('Coeficiente')
    plt.title('Impacto de las Variables en el Modelo GLM (sin constante)')
    plt.axvline(0, color='black', linewidth=1, linestyle='--')  # Línea de referencia en 0
    
    # Mostrar el gráfico
    plt.show()

