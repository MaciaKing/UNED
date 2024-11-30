import pandas as pd
import pdb
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


#-----------------------------------------------
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