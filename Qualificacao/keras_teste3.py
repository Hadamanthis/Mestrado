from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
from keras import backend as K
from readDataset import readDataset

import tensorflow
import theano
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import preprocessing

K.set_image_dim_ordering('th')

n_frames = 30
# image specification
img_rows, img_cols, img_depth = 48, 27, n_frames
resizeShape = (img_rows, img_cols)
interpolation = cv2.INTER_AREA # interpolação utilizada no resize
color = cv2.COLOR_BGR2GRAY # Trocar cor

# Training data
X_tr=[] # variable to store entire dataset

fps = 30

path = '/home/geovane/Mestrado/lsa64_cut1'

# Lendo o dataset
X_tr, labels = readDataset(path, img_depth, color, resizeShape, interpolation)

num_samples = len(X_tr)
print('num_samples:', num_samples)

X = [] # Guarda todas as sub-imagens
	
n_sub_imgs = 1

# Extraindo as imagens da base
for img in X_tr:
	X.append(img) # niveis de cinza
	#X.append( np.absolute(cv2.Sobel(img,cv2.CV_16U,1,0,ksize=5)) ) # gradiente x
	#X.append( np.absolute(cv2.Sobel(img,cv2.CV_16U,0,1,ksize=5)) ) # gradiente y
	#treino_optflwx.append(  )
	#treino_optflwy.append(  )

# convert the frames read into array
X = np.array(X)
labels = np.array(labels)

print('X shape:', X.shape)
print('labels shape:', labels.shape)

TRAIN_TEST = 0.3

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=TRAIN_TEST)

# Depois que divido, tenho que aumentar os casos de teste com o Augmentor
# Porém não quero ter que ler dos arquivos, quero criar em tempo de execução nosvas imagens e só adicionar
# Criar novos vídeos a partir de vídeos da base inicial
#gauss = np.random.normal(0, 0.05,(img_rows, img_cols, 1))
#gauss = gauss.reshape(img_rows, img_cols, 1)
#c1, c2, c3, c4 = [], [], [], []
#for train_img in X_train:
	
	# Flip horizontal
	#dst = cv2.flip(train_img, 0)
	#c1.append(dst)

	# Flip vertical
	#dst = cv2.flip(train_img, 1)
	#c2.append(dst)

	# Equalização de histograma
	#s = []
	#for train_slice in cv2.split(train_img):
	#	s.append(cv2.equalizeHist(train_slice))
	#dst = cv2.merge(s)
	#c3.append(dst)

	# Adicionando ruido gaussiano
	#dst = train_img + gauss
	#c4.append(dst)

#X_train = np.array(X_train.tolist() + c1 + c2 + c3 + c4)
#y_train = np.array(5 * y_train.tolist());

#X_train = np.array(X_train.tolist())
#y_train = np.array(y_train.tolist());

# input_shape(qtd_imagens, qtd_canais, qtd_linhas, qtd_colunas, qtd_profundidade)
TRAIN_SAMPLES = y_train.shape[0]
TEST_SAMPLES = y_test.shape[0]

print(TRAIN_SAMPLES)
print(TEST_SAMPLES)

input_shape = (n_sub_imgs, img_rows, img_cols, n_frames)
train_set = np.zeros((TRAIN_SAMPLES, n_sub_imgs, img_rows, img_cols, n_frames))
test_set = np.zeros((TEST_SAMPLES, n_sub_imgs, img_rows, img_cols, n_frames))
#train_set = np.zeros((num_samples, 1, img_rows, img_cols, img_depth))

h = 0
g = 0
for h in list(range(TRAIN_SAMPLES)):
	#for r in list(range(n_sub_imgs)):
		#train_set[h][r][:][:][:] = X_train[g+r,:,:,:]
	#g += n_sub_imgs
	train_set[h][0][:][:][:] = X_train[h,:,:,:]

h = 0
g = 0
for h in list(range(TEST_SAMPLES)):
	#for r in list(range(n_sub_imgs)):
		#test_set[h][r][:][:][:] = X_test[g+r,:,:,:]
	#g += n_sub_imgs
	test_set[h][0][:][:][:] = X_test[h,:,:,:]

print(train_set.shape, 'train samples')
print(test_set.shape, 'test samples')

# CNN Training parameters
batch_size = 16
nb_classes = 2
nb_epoch = 10

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_train.shape)

# Number of convolutional filter to use at each layer
nb_filters = [32, 32]

# Level of pooling to perform at each layer (POOL x POLL)
nb_pool = [3, 3]

# Level of convolution to perform at each layer (CONV x CONV)
nb_conv = [3, 3]

# Pre-processing 
train_set = train_set.astype('float32')
test_set = test_set.astype('float32')

train_set -= np.mean(train_set)
test_set -= np.mean(test_set)

train_set /= np.max(train_set)
test_set /= np.max(test_set)

# Define model
model = Sequential()

model.add(Convolution3D(nb_filters[0],
 						(nb_conv[0], nb_conv[0], nb_conv[0]),
						input_shape=input_shape,
						activation='relu'))

model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0]), strides=(2, 2, 2)))

model.add(Convolution3D(nb_filters[1],
 						(nb_conv[1], nb_conv[1], nb_conv[1]),
						activation='relu'))

model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0]), strides=(2, 2, 2)))

model.add(Convolution3D(nb_filters[1],
 						(nb_conv[1], nb_conv[1], nb_conv[1]),
						activation='relu'))

model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0]), strides=(2, 2, 2)))

model.add(Flatten())

model.add(Dense(256, activation='relu', kernel_initializer='normal'))

model.add(Dense(128, activation='relu', kernel_initializer='normal'))

model.add(Dense(64, activation='relu', kernel_initializer='normal'))

model.add(Dense(nb_classes, kernel_initializer='normal'))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# Train the model
print('train_set.shape:', train_set.shape)
print('Y_train.shape:', Y_train.shape)
print('test_set.shape:', test_set.shape)
print('Y_test.shape:', Y_test.shape)

hist = model.fit(train_set, Y_train, validation_data=(test_set, Y_test), epochs=nb_epoch, shuffle=True, verbose=1)

# fits the model on batches with real-time data augmentation:
#hist = model.fit_generator(datagen.flow(train_set, Y_train, batch_size=batch_size), steps_per_epoch=len(train_set) / batch_size, epochs=nb_epochs)

# Evaluate the model
#score = model.evaluate(test_set, Y_test)
score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size)
print('\nTest score:', score[0])
print('Test accuracy:', score[1])

# Para obter matriz de confusão
from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(test_set)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

target_names = list(map(str, list(range(1, nb_classes+1)))) # Cria uma lista de inteiros e depois faz um map para uma lista de string
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
confusionMatrix = np.matrix(confusion_matrix(np.argmax(Y_test,axis=1), y_pred)).tofile("output.txt", sep="\t", format="%s")

#model.summary()
