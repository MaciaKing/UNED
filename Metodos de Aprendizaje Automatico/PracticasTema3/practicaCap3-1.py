import pandas as pd
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

# Mostrar el soporte para cada genero
print("Soporte de cada género:")
print(support)
