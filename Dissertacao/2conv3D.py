import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from readDataset import readDataset

from keras import backend as K
K.set_image_dim_ordering('th')

# Parâmetros
DATASET_PATH = '/mnt/DADOS/Bases/lsa64_cut1'
N_ROWS, N_COLS, N_FRAMES = 64, 64, 30
N_CLASSES = 2
TEST_FACTOR = 0.2
N_EPOCHS = 100
N_FILTERS = [32, 64]
N_POOL = [3, 3]
N_CONV = [3, 3]
N_BATCH = 32

# Lendo a base de videos
print("> Lendo a base de videos.")
X, labels = readDataset(DATASET_PATH, N_FRAMES, resizeShape=(N_ROWS, N_COLS))
print("> {0} videos lidos".format(len(X))) 

# Transformando em array numpy
X = np.array(X)
labels = np.array(labels)

# Dividindo a base em treino e teste
print("> Dividindo a base em {0}% treino e {1}% teste".format((1-TEST_FACTOR)*100, TEST_FACTOR*100))
X_train, X_test, label_train, label_test = train_test_split(X, labels, test_size=TEST_FACTOR)
train_len, test_len = label_train.shape[0], label_test.shape[0]

# Convertendo os labels para matriz de classes binárias
print("> Transformando vetor de classes em uma matriz de classes binárias")
label_train = np_utils.to_categorical(label_train, N_CLASSES)
label_test = np_utils.to_categorical(label_test, N_CLASSES)

# Possivel Augmentation
pass

# Definindo as dimensões da entrada da rede
input_shape = (1, N_ROWS, N_COLS, N_FRAMES)
train_set = np.zeros((train_len, 1, N_ROWS, N_COLS, N_FRAMES))
test_set = np.zeros((test_len, 1, N_ROWS, N_COLS, N_FRAMES))

# Redimensionando a entrada do treino
print("> Redimensionando a base de treino")
for h in list(range(train_len)):
	train_set[h][0][:][:][:] = X_train[h, :, :, :]

print("> Base de treino redimensionada para", train_set.shape)

# Redimensionando a entrada do teste
print("> Redimensionando a base de teste")
for h in list(range(test_len)):
	test_set[h][0][:][:][:] = X_test[h, :, :, :]

print("> Base de teste redimensionada para", test_set.shape)

# Preprocessamento
train_set = train_set.astype('float32')
test_set = test_set.astype('float32')

train_set -= np.mean(train_set)
test_set -= np.mean(test_set)

train_set /= np.max(train_set)
test_set /= np.max(test_set)

# Definindo o modelo da rede
print("> Arquitetura da rede sendo construida")
model = Sequential()

model.add(Convolution3D(N_FILTERS[0], (N_CONV[0], N_CONV[0], N_CONV[0]), activation='relu', input_shape=input_shape))

model.add(MaxPooling3D(pool_size=(N_POOL[0], N_POOL[0], N_POOL[0]), strides=(2, 2, 1)))

model.add(Dropout(0.5))

model.add(Convolution3D(N_FILTERS[1], (N_CONV[1], N_CONV[1], N_CONV[1]), activation='relu'))

model.add(Flatten())

#model.add(Dense(128, activation='relu', kernel_initializer='normal'))

model.add(Dense(N_CLASSES, kernel_initializer='normal'))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="RMSprop", metrics=['accuracy'])

# Treinando o modelo
print("> Iniciando o treinamento do modelo...")
hist = model.fit(train_set, label_train, validation_data=(test_set, label_test), epochs=N_EPOCHS, shuffle=True, verbose=1)

# Avaliando o modelo
#score = model.evaluate(test_set, label_test, batch_size=N_BATCH)
score = model.evaluate(test_set, label_test)
print('\nTest score:', score[0])
print('Test accuracy:', score[1])

# Plot the results
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(100)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print(plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()

