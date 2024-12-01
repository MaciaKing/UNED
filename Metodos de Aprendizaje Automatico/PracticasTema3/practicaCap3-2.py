import pandas as pd
import itertools
import time
import pdb
from math import comb
import json
from mlxtend.preprocessing import TransactionEncoder

# Obtenemos los valores de cada columna
genre_names = pd.read_csv('films_dataset/categories.csv', sep='|', header=None, names=['category','category_id'])

# Declaramos los nombres de todas las columnas
df_col_names = ['title', 'release_date', 'release_video_date', 'url'] + list(genre_names['category'].values)
initial_df = pd.read_csv('films_dataset/films.csv', sep='|', header=None, names=df_col_names)

# Declaramos que el indice sea la columna titulo
initial_df.set_index("title", inplace=True)

# Convertimos los 1 y 0 en booleanos
df = initial_df[list(genre_names['category'].values)].astype(bool)

# Convertimos el DataFrame a una lista de transacciones
transactions = df.apply(lambda row: row[row].index.tolist(), axis=1).tolist()

# Instanciar el TransactionEncoder
encoder = TransactionEncoder().fit(transactions)

# One-hot encode las transacciones
onehot = encoder.transform(transactions)

# Convertimos a DataFrame
onehot_df = pd.DataFrame(onehot, columns=encoder.columns_)

# Calculamos el soporte (proporción de 1's por columna)
support = onehot_df.mean()


supportRomance = onehot_df['Romance'].mean()
supportDrama = onehot_df['Drama'].mean()
supportAction = onehot_df['Action'].mean()

supportAdventure = onehot_df['Adventure'].mean()
supportCrime = onehot_df['Crime'].mean()

supportChildrens = onehot_df['Children\'s'].mean()


#-----------------------------------------------------------------------------------
#
# 2. Calcular el soporte, confianza y lift de las siguientes reglas:  
# Romance -> Drama; Action, Adventure -> Thriller; Crime, Action -> Thriller; 
# Crime -> Action, Thriller; Crime -> Children's

def calculate_support_individual(column_name, total_transactions):
    total_transaction_column_name = onehot_df[column_name].sum()  # Número de transacciones donde el género está presente
    return total_transaction_column_name / total_transactions

# Función para calcular soporte conjunto
def calculate_joint_support(columns, total_transactions):
    total_joint_transactions = onehot_df[columns].all(axis=1).sum()  # Transacciones donde todos los géneros están presentes
    return total_joint_transactions / total_transactions

# Función para calcular confianza
def calculate_confidence(antecedents, consequents, total_transactions):
    support_antecedents = calculate_joint_support(antecedents, total_transactions)  # Soporte del antecedente
    support_joint = calculate_joint_support(antecedents + consequents, total_transactions)  # Soporte conjunto
    return support_joint / support_antecedents if support_antecedents > 0 else 0

# Función para calcular lift
def calculate_lift(antecedents, consequents, total_transactions):
    confidence = calculate_confidence(antecedents, consequents, total_transactions)
    support_consequents = calculate_joint_support(consequents, total_transactions)  # Soporte del consecuente
    return confidence / support_consequents if support_consequents > 0 else 0

total_transactions = len(transactions)  # Total de transacciones
# Romance -> Drama
print("Romance -> Drama")
print("Soporte conjunto:", calculate_joint_support(['Romance', 'Drama'], total_transactions))
print("Confianza:", calculate_confidence(['Romance'], ['Drama'], total_transactions))
print("Lift:", calculate_lift(['Romance'], ['Drama'], total_transactions))

# Adventure -> Thriller
print("\nAdventure -> Thriller")
print("Soporte conjunto:", calculate_joint_support(['Adventure', 'Thriller'], total_transactions))
print("Confianza:", calculate_confidence(['Adventure'], ['Thriller'], total_transactions))
print("Lift:", calculate_lift(['Adventure'], ['Thriller'], total_transactions))

print("\nCrime, Action -> Thriller")
# Crime, Action -> Thriller
print("Soporte conjunto:", calculate_joint_support(['Crime', 'Action', 'Thriller'], total_transactions))
print("Confianza:", calculate_confidence(['Crime', 'Action'], ['Thriller'], total_transactions))
print("Lift:", calculate_lift(['Crime', 'Action'], ['Thriller'], total_transactions))

print("\nCrime -> Action, Thriller")
# Crime -> Action, Thriller
print("Soporte conjunto:", calculate_joint_support(['Crime', 'Action', 'Thriller'], total_transactions))
print("Confianza:", calculate_confidence(['Crime'], ['Action', 'Thriller'], total_transactions))
print("Lift:", calculate_lift(['Crime'], ['Action', 'Thriller'], total_transactions))

print("\nCrime -> Children's")
# Crime -> Children's
print("Soporte conjunto:", calculate_joint_support(['Crime', "Children's"], total_transactions))
print("Confianza:", calculate_confidence(['Crime'], ["Children's"], total_transactions))
print("Lift:", calculate_lift(['Crime'], ["Children's"], total_transactions))

# 3. Discutid y justificad en un comentario del código qué opináis 
# de los resultados de soporte, confianza y lift de estas reglas.
# 
# ** Romance -> Drama:
#    Support: 0.058858501783590964
#    Confianza: 0.4008097165991903
#    Lift: 0.9298785425101216
#         El Support es de 5.89% de las transacciones contienen tanto 
#         Romance como Drama. Este valor sugiere que hay una relación moderada entre estos 
#         dos géneros, ya que no es un valor extremadamente bajo, pero tampoco es muy alto. 
#         La confianza lo que indica que el 40.08% de las transacciones con 
#         Romance también incluyen Drama. Esto muestra que, si una película está clasificada 
#         como Romance, hay una probabilidad considerable de que también sea clasificada como Drama.
#         El lift nos indica que la relación entre Romance y Drama no es particularmente fuerte.
#
# ** Adventure -> Thriller:
#    Support: 0.01248513674197384
#    Confianza: 0.15555555555555556
#    Lift: 1.0424081451969898
#      El support sugiere que esta combinación es relativamente rara en el conjunto de datos. La confianza 
#      es de 15.56% de las transacciones con Adventure también contienen Thriller. Aunque esta cifra 
#      no es alta, muestra que hay una relación moderada entre ambos géneros. Sin embargo, el lift es de 
#      1.0424, lo que indica que existe una ligera asociación positiva entre Adventure y Thriller.
# 
# ** Crime, Action -> Thriller:
#    Support: 0.0023781212841854932
#    Confianza: 0.17391304347826086
#    Lift: 1.1654252554997402
#       El support refleja que la combinación de estos tres géneros es bastante rara. La confianza es de 0.1739, 
#       lo que indica que el 17.39% de las transacciones que contienen Crime y Action también incluyen Thriller. 
#       Aunque no es una relación muy fuerte, existe una tendencia a que estos géneros aparezcan juntos. 
#       El lift es de 1.1654, lo que sugiere una relación positiva entre Crime, Action y Thriller. A pesar de que la confianza no 
#       es alta, el lift mayor a 1 indica que la combinación de estos géneros ocurre con más frecuencia de 
#       lo que sería esperado si fueran independientes.
#    
# ** Crime -> Action, Thriller
#    Soporte conjunto: 0.0023781212841854932
#    Confianza: 0.03669724770642202
#    Lift: 0.7348186981214504
#       El support es del 0.24%. Este valor refleja que la combinación de estos tres géneros 
#       es bastante rara. La confianza es de 0.1739, lo que indica que el 17.39% de las transacciones que contienen 
#       Crime y Action también incluyen Thriller. Aunque no es una relación muy fuerte, existe una cierta tendencia 
#       a que estos géneros aparezcan juntos. El lift es de 1.1654, lo que sugiere una relación positiva entre 
#       Crime, Action y Thriller. A pesar de que la confianza no es alta, el lift mayor a 1 indica que la 
#       combinación de estos géneros ocurre con más frecuencia de lo que sería esperado si fueran independientes.
#
# ** Crime -> Children's
#    Support: 0.0
#    Confianza: 0.0
#    Lift: 0.0
#       Este support es de 0.0, lo que indica que no hay transacciones que contengan tanto Crime 
#       como Children's. La confianza también es 0, ya que no hay transacciones que contengan Crime y 
#       Children's al mismo tiempo. Finalmente, el lift es también 0, deducimos que no hay relacion.
#
#
# 4. Ahora vamos a generalizar este asunto: crearemos una función genérica "rule_metrics" que admita dos 
# listas de strings (la lista de genéros del  antecedente y la lista de géneros del consecuente de la de la regla) 
# junto con el dataset onehot. La función debe devolver como resultado un diccionario con 4 elementos: una cadena de 
# la forma  "A, B, ... -> C, D, ..." y tres floats que con sus resultados de support, confidence and lift. 
# Ejemplo: rule_metrics(["Action","Adventure"],["Thriller"],onehot_dataset) devolvería algo como 
# {"rule": "Action, Adventure -> Thriller", "support": 0.12, "confidence": 0.2, "lift": 1.3}  
# (los valores son un ejemplo, no son los reales)
#
#
print("\n\nEjercicio 4")
def rule_metric(antecedent_list, consequent_list, onehot_df):
    rule_string = ', '.join(antecedent_list)
    rule_string = rule_string + " -> "
    rule_string = rule_string + ', '.join(consequent_list)
    support = calculate_joint_support(antecedent_list + consequent_list, len(onehot_df))
    confidence = calculate_confidence(antecedent_list, consequent_list, len(onehot_df))
    lift = calculate_lift(antecedent_list, consequent_list, total_transactions)
    return {"rule": rule_string, "support": support, "confidence": confidence, "lift": lift}



json_example = rule_metric(["Action","Adventure"],["Thriller"], onehot_df)
print(json_example)

#
# 5. Calcular matemáticamente y de manera justificada cuantas reglas de tipo 
# A -> B se pueden construir para este dataset.
#
# Total reglas = (2^(19)−1)⋅(2^(19)−1) = 524,287 ⋅ 524,287 = 274,877,906,944.


# 6. Mediante la función "rule_metrics", generar un DataFrame con todas las reglas 
# de tipo A -> B con rule, support, confidence y lift como columnas. 
# ¿Cuánto tiempo ha tardado vuestro código en generarlas?
# No llega a acabar el codigo ya que el computo computacional es muy elevado

# Filtros mínimos
MIN_SUPPORT = 0.01  # 1% de soporte mínimo
MIN_CONFIDENCE = 0.1  # 10% de confianza mínima

# Función para generar reglas optimizadas
def generate_filtered_rules(onehot_df, min_support, min_confidence):
    start_time = time.time()
    rules = []
    genres = list(onehot_df.columns)
    
    # Iterar sobre todos los subconjuntos posibles para A y B
    for i in range(1, len(genres)):  # Evitar conjuntos vacíos
        antecedents = list(itertools.combinations(genres, i))
        for antecedent in antecedents:
            for j in range(1, len(genres) - len(antecedent) + 1):
                consequents = list(itertools.combinations(set(genres) - set(antecedent), j))
                for consequent in consequents:
                    # Calcular métricas
                    metrics = rule_metric(list(antecedent), list(consequent), onehot_df)
                    
                    # Aplicar filtros
                    if metrics['support'] >= min_support and metrics['confidence'] >= min_confidence:
                        rules.append(metrics)
    
    end_time = time.time()
    print(f"Tiempo total: {end_time - start_time:.2f} segundos")
    return pd.DataFrame(rules)

# Ejecutar generación de reglas con filtros
# filtered_rules_df = generate_filtered_rules(onehot_df, MIN_SUPPORT, MIN_CONFIDENCE)


#
# 7. Ordenar de mayor a menor por support, confidence, lift 
# (por este orden) y mostrar las 20 primeras reglas anteriores 
# (junto con sus valores de support, confidence, lift) como comentario.
#


# 8. Calcular matemáticamente y de manera justificada 
# cuantas reglas de tipo A,B -> C ó A -> B,C se pueden construir para este dataset.
#
# El dataset tiene n = 19 géneros.
#    1) A,B --> C: Donde A,B son antecedentes, y C es un único consecuente.
#    2) A --> B,C: Donde A es el antecedente, y B, C son consecuentes.
#
# 1) Calculo para las reglas A,B --> C
#    Total combinaciones para los antecedentes
#       (19 sobre 2) = (19 ⋅ 18)/2 = 171
#
#    Total combinaciones para los consequentes
#        (17 sobre 1) = 17
#
#    Total reglas A,B --> C: (19 sobre 2) ⋅ (17 sobre 1) = 171 ⋅ 17 = 2907
#
#
# 2) Calculo para las reglas A --> B,C
#    Total combinaciones para A.
#       (19 sobre 1) = 19
# 
#    Total combinaciones para los consequentes.
#       (18 sobre 2) = (18 ⋅ 17)/2 = 153
#
#    Total reglas A --> B,C:
#       (19 sobre 1) ⋅ (18 sobre 2) = 19 ⋅ 153 = 2907
# 
# Total reglas = 2907 + 2907 = 5814
#


# 9. Mediante la función "rule_metrics", generar un DataFrame con todas las 
# reglas de tipo A,B -> C ó A -> B,C con rule, support, confidence y lift 
# como columnas. ¿Cuánto tiempo ha tardado vuestro código en generarlas?
#
# Este me llega a tardar una media de 6 segundos en generar las reglas.


def generate_AB_to_C_and_A_to_BC(onehot_df):
    start_time = time.time()  # Registrar el inicio
    rules = []
    genres = list(onehot_df.columns)
    
    # 1. Reglas de tipo A, B -> C
    for antecedents in itertools.combinations(genres, 2):  # Elegir 2 géneros como antecedentes
        for consequent in set(genres) - set(antecedents):  # Elegir 1 género como consecuente
            metrics = rule_metric(list(antecedents), [consequent], onehot_df)
            rules.append(metrics)
    
    # 2. Reglas de tipo A -> B, C
    for antecedent in genres:  # Elegir 1 género como antecedente
        for consequents in itertools.combinations(set(genres) - {antecedent}, 2):  # Elegir 2 géneros como consecuentes
            metrics = rule_metric([antecedent], list(consequents), onehot_df)
            rules.append(metrics)
    
    end_time = time.time()  # Registrar el final
    print(f"Tiempo total: {end_time - start_time:.2f} segundos")
    
    return pd.DataFrame(rules)

# Generar las reglas y medir el tiempo
rules_df = generate_AB_to_C_and_A_to_BC(onehot_df)


# 10. Ordenar de mayor a menor por support, confidence, lift (por este orden)
# y mostrar las 20 primeras reglas anteriores (junto con sus valores de support, 
# confidence, lift) como comentario.
sorted_rules_df = rules_df.sort_values(by=["support", "confidence", "lift"], ascending=False)
print(sorted_rules_df.head(20))


# 11. Calcular matemáticamente y de manera justificada cuantas reglas de 9 
# elementos se pueden construir para este dataset.
# 
# Total reglas de 9 elementos = 47,112,780

# Número de combinaciones para seleccionar 9 elementos de 19
num_combinations_9 = comb(19, 9)

# Número de formas de dividir esos 9 elementos en antecedentes y consecuentes
num_divisions = (2**9) - 2  # Restamos los casos vacíos

# Total de reglas
total_rules_9 = num_combinations_9 * num_divisions
print(f"Total de reglas de 9 elementos: {total_rules_9}")

# 12. Considerando que tardaríais una proporción similar de tiempo que en las 
# anteriores ¿cuánto tiempo tardaría vuestro código en generar todas estas reglas de 9 elementos?
#
# Si para 5814 ha tardado 6 minutos
#  5814 -- > 6m
#  7,112,780 -- > x
#
#  x = (7,112,780 * 6m) / 5814 = 7340.33 m  --> 12233,88333 horas
# No acabaria nunca

# 13. Calcular matemáticamente y de manera justificada cuantas reglas de todo 
# tipo que contengan desde 1 hasta 19 elementos (todas las reglas posibles) se pueden 
# construir para este dataset. Siguiendo el enfoque anterior ¿cuánto tiempo tardaría 
# vuestro código en generarlas?
#
# Calcular el total de reglas de tipo A -> B para 1 a 19 elementos
# Para el primer caso = (19 sobre 1) ⋅ (2^1 - 2) = 0
# Para el segundo caso = (19 sobre 2) ⋅ (2^2 - 2) = 342
# Para el tercer caso = (19 sobre 3) ⋅ (2^3 - 2) = 0
# ...
# Para el ultimo caso = (19 sobre 19) ⋅ (2^19 - 2) = 524286
#
# Suma de todas las reglas = 0+ 342 + 5814 + 54264 + 348840 + 
# 1684194 + 6348888 + 19166428 + 47112780 + 94403916 + 188711508 + 
# 206533032+222232308 + 190120536 + 127338261 + 63515826 + 
# 22427670 + 4980748 + 524286 = 1195509641

total_rules_all = sum(comb(19, k) * (2**k - 2) for k in range(1, 20))
print(f"Total de reglas de 1 a 19 elementos: {total_rules_all}")