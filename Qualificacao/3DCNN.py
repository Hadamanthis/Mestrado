from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
from keras import backend as K

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

def read_class(X_tr, folder, fps, n_frames, resizeShape = None,
 interpolation = cv2.INTER_AREA, cvtColor = cv2.COLOR_BGR2GRAY):
	
	listing = os.listdir(folder)
	
	for vid in listing:
		vid = folder + '/' + vid
		frames = []
		cap = cv2.VideoCapture(vid)
		print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
		
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

		#print(input.shape)
		ipt = np.rollaxis(np.rollaxis(input, 2, 0), 2, 0)
		#print(ipt.shape)
		
		X_tr.append(ipt)

	return X_tr

# image specification
img_rows, img_cols, img_depth=16, 16, 15
resizeShape = (img_rows, img_cols)

# Training data
X_tr=[] # variable to store entire dataset

fps = 25
n_frames = 15

path = "/home/geovane/Transferências/kth dataset"

# Reading boxing action class
X_tr = read_class(X_tr, path + '/boxing', fps, n_frames, resizeShape)

# Reading hand clapping action class
X_tr = read_class(X_tr, path + '/handclapping', fps, n_frames, resizeShape)

# Reading hand waving action class
X_tr = read_class(X_tr, path + '/handwaving', fps, n_frames, resizeShape)

# Reading jogging action class
X_tr = read_class(X_tr, path + '/jogging', fps, n_frames, resizeShape)

# Reading running action class
X_tr = read_class(X_tr, path + '/running', fps, n_frames, resizeShape)

# Reading walking action class
X_tr = read_class(X_tr, path + '/walking', fps, n_frames, resizeShape)

# convert the frames read into array
X_tr_array = np.array(X_tr)

num_samples = len(X_tr_array)
print('num_samples:', num_samples)

# Assign Label to each class

label = np.ones((num_samples), dtype = int)
label[0:100] = 0
label[100:199] = 1
label[199:299] = 2
label[299:399] = 3
label[399:499] = 4
label[499:] = 5

train_data = [X_tr_array, label]

(X_train, y_train) = (train_data[0], train_data[1])
print('X_Train shape:', X_train.shape)

train_set = np.zeros((num_samples, 1, img_rows, img_cols, img_depth))

print(X_train.shape)

for h in list(range(num_samples)):
	train_set[h][0][:][:][:] = X_train[h,:,:,:]

print(train_set.shape, 'train samples')

# CNN Training parameters
batch_size = 2
nb_classes = 6
nb_epoch = 50

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)

# Number of convolutional filter to use at each layer
nb_filters = [32, 32]

# Level of pooling to perform at each layer (POOL x POLL)
nb_pool = [3, 3]

# Level of convolution to perform at each layer (CONV x CONV)
nb_conv = [5, 5]

# Pre-processing 
train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /= np.max(train_set)

# Define model
model = Sequential()

model.add(Convolution3D(nb_filters[0], (nb_conv[0], nb_conv[0], nb_conv[0]), input_shape=(1, img_rows, img_cols, img_depth), activation='relu'))

model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu', kernel_initializer='normal'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes, kernel_initializer='normal'))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# Split the data
X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_set, Y_train, test_size=0.2, random_state=4)

# Train the model
hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new, y_val_new), batch_size=batch_size, epochs=nb_epoch, shuffle=True)

# Evaluate the model
score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size)
print('\nTest score:', score[0])
print('Test accuracy:', score[1])

# Plot the results
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']
xc = list(range(50))

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train', 'val'])
#print(plt.style.available) # use bmh, classic, ggplot for big pictures
plt.style.use(['classic'])
plt.show()

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
#print(plt.style.available) # use bmh, classic, ggplot for big pictures
plt.style.use(['classic'])
plt.show()
