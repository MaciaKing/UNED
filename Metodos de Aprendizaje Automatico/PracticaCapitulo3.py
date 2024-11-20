# Make Predictions with Naive Bayes On The Iris Dataset
from random import seed, randrange
from math import sqrt
from math import exp
from math import pi, log
from ucimlrepo import fetch_ucirepo
import pandas as pd
import pdb
 
# Convierte una columna de cadenas de texto (string) a números flotantes (float)
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column])

# Convertir columna de cadenas de texto a números flotantes
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column])
 
# Convertir una columna de cadenas de texto a números enteros
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 
# Dividir un conjunto de datos en k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calcular el porcentaje de precisión (accuracy)
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluar un algoritmo utilizando validación cruzada (cross-validation)
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
 
# Dividir el conjunto de datos por valores de clase y devuelve un diccionario
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated
 
# Calcular la media de una lista de números
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
# Calcular la desviación estándar de una lista de números
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
 
# Calcular la media, desviación estándar y 
# cantidad de elementos para cada columna de un conjunto de dato
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries
 
# Dividir el conjunto de datos por clase y luego calcular estadísticas para cada fila
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries
 
# Calcular la función de distribución de probabilidad gaussiana para x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calcular la función de distribución de probabilidad logarítmica para x
def calculate_log_probability(x, mean, stdev):
    exponent = -((x - mean) ** 2) / (2 * stdev ** 2)
    log_prob = exponent - log(sqrt(2 * pi) * stdev)
    return log_prob
 
# Calcular las probabilidades de predecir cada clase para una fila dada
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			#probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
			probabilities[class_value] *= calculate_log_probability(row[i], mean, stdev)
	return probabilities
 
# Predecir la clase para una fila dada
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

# Vamos a identificar qué columnas de nuestro DataFrame pueden ser categóricas.
# Realizaremos un recorrido por el DataFrame para ver su descripción.
# Si encontramos que la salida del método describe contiene "unique", significa que los valores son únicos y potencialmente categóricos.
def convert_columns_to_categorical_data(df):
    print("**** Tipos de datos ANTES de convertir a categóricos:\n")
    print(df.dtypes)
    
    for (nombre_columna, datos_columna) in df.items():
        descripcion = df[nombre_columna].describe()    
        # Si la columna tiene valores únicos, la convertimos a tipo categórico.
        if "unique" in descripcion.index:
            df[nombre_columna] = df[nombre_columna].astype('category')

    print("**** Tipos de datos DESPUÉS de convertir a categóricos:\n")
    print(df.dtypes)

# Convertir columnas categóricas a valores enteros (códigos)
def convert_categorical_to_int(df):
    # Recorremos todas las columnas que son de tipo 'category'
    for col in df.select_dtypes(['category']).columns:
        # Convertimos los valores categóricos a sus respectivos códigos enteros
        df[col] = df[col].cat.codes

# Algoritmo de Naive Bayes
def naive_bayes(train, test):
	summarize = summarize_by_class(train)
	predictions = list()
	for row in test:
		output = predict(summarize, row)
		predictions.append(output)
	return(predictions)

# Test Naive Bayes on Iris Dataset
##### Iris Dataset #####
seed(1)
dataset = fetch_ucirepo(id=53)
df_total = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
dataset = df_total.values.tolist()
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

##### Bank marketing #####
bank_marketing = fetch_ucirepo(id=222)
df_total = pd.concat([bank_marketing.data.features, bank_marketing.data.targets], axis=1)

# Eliminamos todas las filas que contengan valores nans
df_total = df_total.dropna()

# Convertimos las posibles variables a category.
convert_columns_to_categorical_data(df_total)
# Convertimos las category variables a numeros.
convert_categorical_to_int(df_total)

last_line = df_total.iloc[-1]
# Elimina la última fila del DataFrame original

df_total = df_total.iloc[:-1]

dataset = df_total.values.tolist() 

for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)

# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# fit model
model = summarize_by_class(dataset)

# evaluate algorithm
n_folds = 5
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

# define a new record
last_line = last_line.values.tolist()
last_line_2 = df_total.iloc[-4]
last_line_2 = last_line_2.values.tolist()


label = predict(model, last_line)
print('Data=%s, Predicted: %s' % (last_line, label))

label = predict(model, last_line_2)
print('Data=%s, Predicted: %s' % (last_line_2, label))