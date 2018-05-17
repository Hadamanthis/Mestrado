import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras import backend as K
from readDataset import readDataset

folder = '/home/geovane/Imagens/LISA_HG_Data/dynamic_gestures/data' # Diretório do dataset
n_classes = 19 # numero de classes
n_epochs = 50 # numero de épocas
color = cv2.COLOR_BGR2GRAY # Trocar cor
n_rows, n_cols, n_frames = 16, 16, 6 #80, 30, 6 # Linhas, Colunas e Frames por video
resizeShape = (n_rows, n_cols) # dimensão pretendida dos frames
interpolation = cv2.INTER_AREA # interpolação utilizada no resize

if __name__ == "__main__":
	# Mudar ordem de linhas e colunas
	K.set_image_dim_ordering('th')

	# Dados de treinamento
	inputs = [] # variavel pra guardar todas as entradas
	labels = [] # Labels das classes

	# Lendo o dataset
	inputs, labels = readDataset(folder, n_frames, color, resizeShape, interpolation)

	num_samples = len(inputs)

	print('> Base de imagens carregada')
	print('> Total de imagens: ', num_samples)

	X = [] # Guarda todas as sub-imagens
	n_sub_imagens = 1 # gray

	# Extraindo as imagens da base
	for img in inputs:
		X.append(img) # niveis de cinza
		#X.append( np.absolute(cv2.Sobel(img,cv2.CV_16U,1,0,ksize=5)) ) # gradiente x
		#X.append( np.absolute(cv2.Sobel(img,cv2.CV_16U,0,1,ksize=5)) ) # gradiente y
		#treino_optflwx.append(  )
		#treino_optflwy.append(  )

	#print('> Sub imagens extraidas')

	# Transformando entrada e labels em array Numpy
	X = np.array(X)
	Y = np.array(labels)

	print('[*] Formato de X:', X.shape)
	
	# Transformando o vetor de classes em matrizes de classes binárias
	Y = np_utils.to_categorical(Y, n_classes)

	# input_shape(qtd_imagens, qtd_canais, qtd_linhas, qtd_colunas, qtd_profundidade)
	input_shape = (n_sub_imagens,
				 n_cols,
				 n_rows,
				 n_frames)

	# treinamento no formato (numero de videos, input_shape)
	train_set = np.zeros((num_samples,
				 		n_sub_imagens,
						n_cols,
				 		n_rows,
						n_frames))

	for h in list(range(num_samples)):
		for r in list(range(n_sub_imagens)):
			train_set[h][r][:][:][:] = X[h,:,:,:]

	print('[*] Formato de train_set:', train_set.shape)
	print('[*] Formato de Y:', Y.shape)

	# Pre-processing 
	train_set = train_set.astype('float32')
	train_set -= np.mean(train_set)
	train_set /= np.max(train_set)

	# Separando os dados em treino/teste 80/20
	X_treino, X_teste, Y_treino, Y_teste = train_test_split(train_set, Y,
															test_size = 0.2, random_state = 4)
	print('> Divisao de treino/teste realizada')
	print('[*] Formato de X_treino:', X_treino.shape)
	print('[*] Formato de X_teste:', X_teste.shape)
	print('[*] Formato de Y_treino:', Y_treino.shape)
	print('[*] Formato de Y_teste:', Y_teste.shape)
	
	# Hyper Parâmetros
	# Numero de filtros convolucionais em cada layer
	n_filters = [15]
	# Profundidade de convolução utilizado em cada layer (CONV x CONV)
	n_conv = [3]
	n_width = [5]
	n_height = [5]
	# imagens a serem treinadas por vez
	batch_size = 2

	# Criar modelo
	model = Sequential()

	model.add(Convolution3D(n_filters[0],
							(n_height[0], n_width[0], n_conv[0]),
							input_shape=input_shape,
							activation='relu'))

	model.add(Dropout(0.5))

	model.add(Convolution3D(n_filters[0],
							(n_height[0], n_width[0], n_conv[0]),
							activation='relu'))

	model.add(Dropout(0.5))

	model.add(Flatten())

	model.add(Dense(128, activation='relu', kernel_initializer='normal'))

	model.add(Dropout(0.5))

	model.add(Dense(n_classes, kernel_initializer='normal'))

	model.add(Activation('softmax'))

	# Cada ação é classificada da maneira "one-against-rest"
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

	# Treinando o modelo
	hist = model.fit(X_treino, Y_treino,  validation_data=(X_teste, Y_teste), epochs=n_epochs, shuffle=True, verbose=1)

	# avaliando o modelo o modelo
	score = model.evaluate(X_teste, Y_teste)
	print('\nTest score:', score[0])
	print('Test accuracy', score[1])

	model.summary()



