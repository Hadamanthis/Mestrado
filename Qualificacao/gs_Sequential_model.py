# Retirado do keras.io (Guide to the Sequential model)

from keras.models import Sequential
from keras.layers import Dense, Activation

# Passando as layers na construção do modelo
model = Sequential([Dense(32, input_shape=(784,)),
					Activation('relu'),
					Dense(10),
					Activation('softmax')])

# Ou alternativamente adicionando cada uma via o método .add()
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

# Especificando o formato da entrada na primeira layer do modelo é obrigatorio com o parametro 'input_shape'

# Algumas layers 3D temporais suportam os argumentos input_dim e input_length

# Se você precisar fixar o tamanho do batch das suas entradas (util para stateful recurrent networks), você pode passar o argumento 'batch_size' para a layer. Se você passar ambos 'batch_size=32' e 'input-shape(6, 8)' para uma layer, será esperado que para toda entrada batch de inputs o batch terá o formato (32, 6, 8)

# São
model = Sequential()
model.add(Dense(32, input_shape=(784,)))

# Equivalentes
model = Sequential()
model.add(Dense(32, input_dim=784))

## Compilation ##

# Antes de treinar um modelo, você precisa configurar o processo de aprendizagem, que é feito via método 'compile', que recebe três argumentos:
# -> Um otimizador. Pode ser uma string que identifica um otimizador existente (como 'rmsprop' ou 'adagrad'), ou uma instancia da classe Optimizer.

# -> Uma função de perda. É o objetivo que esse modelo tentará minimizar. Pode ser uma string que identifica uma função de perda existente (como 'categorical_cossentropy' ou 'mse'), ou pode ser uma função objetivo.

# -> Uma lista de métricas. Para qualquer problema de classificação você gostará de adicionar a métrica de acurácia através do argumento "metrics=['accuracy']". Uma métrica pode ser uma string identificada em uma métrica existente ou uma função de métrica customizada

# Para um problema de classificação multiclasse
model.compile(optimizer='rmsprop',
				loss='categorical_crossentropy',
				metrics=['accuracy'])

# Para um problema de classificação binária
model.compile(optimizer='rmsprop',
				loss='binary_crossentropy',
				metrics=['accuracy'])

# Para um problema de regressão de erro quadrático médio
model.compile(optimizer='rmsprop', loss='mse')

# Para uma métrica customizada
import keras.backend as K

def mean_pred(y_true, y_pred):
	return K.mean(y_pred)

model.compile(optmizer='rmsprop',
				loss='binary_crossentropy',
				metrics=['accuracy', mean_pred])

## Training ##

# Os modelos do keras são treinados em arrays numpy de dados de entrada e labels. Para um modelo de treinamento, tipicamente utilizamos a função 'fit'.

#<## 
# Para um modelo de entrada simples com duas classes (classificação binaria):
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
				loss='binary_crossentropy', 
				metrics=['accuracy'])

# Gerando dados dummy
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Treina o modelo, iterando em batches de amostra 32
model.fit(data, labels, epochs=10, batch_size=32)
##>#

#<##
# Para um modelo de entrada simples com 10 classes (classificação multiclasse):
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
				loss='categorical_crossentropy', 
				metrics=['accuracy'])

# Gerando dados dummy
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Convertendo labels para codificação categorica one-hot
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# Treinar o modelo, interando nos dados em batches de amostra 32
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
#>## 














