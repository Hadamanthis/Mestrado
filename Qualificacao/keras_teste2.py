from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
from keras import backend as K
from readDataset import readDataset

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

def read_class(X_tr, folder, n_frames, resizeShape = None,
 interpolation = cv2.INTER_AREA, cvtColor = cv2.COLOR_BGR2GRAY):
	
	listing = os.listdir(folder)
	
	for vid in listing:
		vid = folder + '/' + vid
		frames = []
		cap = cv2.VideoCapture(vid)
		print("Frames captured")
		
		for k in list(range(n_frames)):
			ret, frame = cap.read()
			
			# Resizing
			if (resizeShape != None):
				frame = cv2.resize(frame, (resizeShape[0], resizeShape[1]), interpolation=interpolation)
	
			# Converting color space
			t_frame = cv2.cvtColor(frame, cvtColor)
			frames.append(t_frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			
		cap.release()

		input = np.array(frames)
		
		print('input shape:', input.shape)		
	
		#print(input.shape)
		ipt = np.rollaxis(np.rollaxis(input, 2, 0), 2, 0)
		print('input shape:', ipt.shape)
		#print(ipt.shape)
		
		X_tr.append(ipt)

	return X_tr

# image specification
img_rows, img_cols, img_depth=16, 16, 6
resizeShape = (img_rows, img_cols)
interpolation = cv2.INTER_AREA # interpolação utilizada no resize
color = cv2.COLOR_BGR2GRAY # Trocar cor

# Training data
X_tr=[] # variable to store entire dataset

fps = 25
n_frames = 6

path = '/mnt/DADOS/LISA_HG_Data/dynamic_gestures/data'

# Lendo o dataset
X_tr, labels = readDataset(path, img_depth, color, resizeShape, interpolation)

num_samples = len(X_tr)
print('num_samples:', num_samples)

X = [] # Guarda todas as sub-imagens
	
n_sub_imgs = 3

# Extraindo as imagens da base
for img in X_tr:
	X.append(img) # niveis de cinza
	X.append( np.absolute(cv2.Sobel(img,cv2.CV_16U,1,0,ksize=5)) ) # gradiente x
	X.append( np.absolute(cv2.Sobel(img,cv2.CV_16U,0,1,ksize=5)) ) # gradiente y
	#treino_optflwx.append(  )
	#treino_optflwy.append(  )

# convert the frames read into array
X = np.array(X)
labels = np.array(labels)

print('X.shape:', X.shape)

# Assign Label to each class

label = labels

print(label.shape)

train_data = [X, label]

(X_train, y_train) = (train_data[0], train_data[1])
print('X_Train shape:', X_train.shape)
print('y_train shape:', y_train.shape)

# input_shape(qtd_imagens, qtd_canais, qtd_linhas, qtd_colunas, qtd_profundidade)
input_shape = (n_sub_imgs, img_rows, img_cols, n_frames)
train_set = np.zeros((num_samples, n_sub_imgs, img_rows, img_cols, n_frames))
#train_set = np.zeros((num_samples, 1, img_rows, img_cols, img_depth))

h = 0
g = 0
for h in list(range(num_samples)):
	for r in list(range(n_sub_imgs)):
		train_set[h][r][:][:][:] = X[g+r,:,:,:]	
	g += n_sub_imgs

print(train_set.shape, 'train samples')

# CNN Training parameters
batch_size = 2
nb_classes = 32
nb_epoch = 50

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(label, nb_classes)

print('Y_train shape:', Y_train.shape)

# Number of convolutional filter to use at each layer
nb_filters = [32, 32]

# Level of pooling to perform at each layer (POOL x POLL)
nb_pool = [3, 3]

# Level of convolution to perform at each layer (CONV x CONV)
nb_conv = [3, 3]

# Pre-processing 
train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /= np.max(train_set)

# Define model
model = Sequential()

model.add(Convolution3D(nb_filters[0],
 						(nb_conv[0], nb_conv[0], nb_conv[0]),
						input_shape=input_shape,
						activation='relu'))

model.add(Dropout(0.5))

model.add(Convolution3D(nb_filters[1],
 						(nb_conv[1], nb_conv[1], nb_conv[1]),
						activation='relu'))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(528, activation='relu', kernel_initializer='normal'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes, kernel_initializer='normal'))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# Split the data
X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_set, Y_train, test_size=0.2, random_state=4)

print(X_train_new.shape)

# Train the model

print('X_train_new.shape:', X_train_new.shape)
print('y_train_new.shape:', y_train_new.shape)
print('X_val_new.shape:', X_val_new.shape)
print('y_val_new.shape:', y_val_new.shape)

hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new, y_val_new), epochs=nb_epoch, shuffle=True, verbose=1)
#hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new, y_val_new), batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1)

# Evaluate the model
score = model.evaluate(X_val_new, y_val_new)
#score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size)
print('\nTest score:', score[0])
print('Test accuracy:', score[1])

#model.predict(X_val_new)

model.summary()
