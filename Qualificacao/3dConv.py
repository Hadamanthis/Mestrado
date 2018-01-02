import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras import backend as K
from readDataset import readDataset


# Diretório do dataset
folder = '/mnt/DADOS/LISA_HG_Data/dynamic_gestures/data' # onde estão as imagens
n_classes = 32 # numero de classes
n_epochs = 50 # numero de épocas
resizeShape = (80, 30) # dimensão pretendida dos frames
interpolation = cv2.INTER_AREA # interpolação utilizada no resize
color = cv2.COLOR_BGR2GRAY # Trocar cor

vid_shape = {'rows':50, 'cols':100, 'frames':6} # shape do video


if __name__ == "__main__":

	# Mudar ordem de linhas e colunas
	K.set_image_dim_ordering('th')
	
	#############################################################
	### Ler imagens (250x115) e coloca-las em uma variável X. ###
	### Há uma qtd variavel de frames por vídeo				  ###	
	### Tem 16 usuarios, performando 32 gestos 3 vezes cada   ###
	#############################################################
		
	# Dados de treinamento
	X_tr = [] # variavel pra guardar todas as entradas
	labels = [] # Labels das classes

	# Lendo o dataset
	X_tr, labels = readDataset(folder, vid_shape['frames'], color, resizeShape, interpolation)
	
	if resizeShape != None:
		vid_shape['rows'] = resizeShape[1]
		vid_shape['cols'] = resizeShape[0] 

	###################################################
	### (Hardwired)									###
	### Transformar cada uma das imagens            ###
	### em 5 (nivel de cinza,   			        ###
	###		gradiente_x, gradiente_y,				###
	###		optflow_x, optflow_y) 		 			###
	###################################################
	
	X = [] # Guarda todas as sub-imagens
	
	# Extraindo as imagens da base
	for img in X_tr:
		X.append(img) # niveis de cinza
		X.append( np.absolute(cv2.Sobel(img,cv2.CV_16U,1,0,ksize=5)) ) # gradiente x
		X.append( np.absolute(cv2.Sobel(img,cv2.CV_16U,0,1,ksize=5)) ) # gradiente y
		#treino_optflwx.append(  )
		#treino_optflwy.append(  )
		
	# Transformando entrada e labels em array Numpy
	X = np.array(X)
	Y = np.array(labels)
	num_samples = len(X_tr)
	
	print('X.shape:', X.shape)
	print('Y.shape:', Y.shape)

	# input_shape(qtd_imagens, qtd_canais, qtd_linhas, qtd_colunas, qtd_profundidade)
	input_shape = (3, vid_shape['rows'], vid_shape['cols'], vid_shape['frames'])
	train_set = np.zeros((num_samples, 3, vid_shape['rows'], vid_shape['cols'], vid_shape['frames']))

	for h in list(range(num_samples)):
		for r in list(range(3)):
			train_set[h][r][:][:][:] = X[h,:,:,:]

	print('train_set:', train_set.shape)

	# Pre-processing 
	train_set = train_set.astype('float32')

	train_set -= np.mean(train_set)

	train_set /= np.max(train_set)

	# Transformando o vetor de classes em matrizes de classes binárias
	Y = np_utils.to_categorical(Y, n_classes)

	print('Y.shape:', Y.shape)

	# Separando os dados em treino/teste 80/20
	X_treino, X_teste, Y_treino, Y_teste = train_test_split(train_set, Y,
																	test_size = 0.2,
																	random_state = 4)

	print('X_treino:', X_treino.shape)
	print('X_teste:', X_teste.shape)
	print('Y_treino:', Y_treino.shape)
	print('Y_teste:', Y_teste.shape)
	
	######################################################################
	### Criar modelo												   ###
	### Adicionar uma camada de convolução 3D com 2 filtros (9x7)	   ###
	### Adicionar uma camada de amostragem (3x3)					   ###
	### Adicionar uma camada de convolução 3D com 3 filtros (7x7)	   ###
	### Adicionar uma camada de amostragem (3x3)					   ###
	### Adicionar uma camada de convolução 3D com um 1 filtro (6x4)	   ###
	### Adicionar camada completamente conectada a 128 feature vectors ###
	### Adicionar ultima camada com 32 saidas						   ###
	######################################################################

	# Numero de filtros convolucionais em cada layer
	n_filters = [16, 16]

	# Profundidade de convolução utilizado em cada layer (CONV x CONV)
	n_conv = [3, 3, 3]
	n_width = [51, 79, 30]
	n_height = [21, 35, 12]
	
	# imagens a serem treinadas por vez
	batch_size = 2

	# Criar modelo
	model = Sequential()

	model.add(Convolution3D(n_filters[0],
							(n_height[0], n_width[0], n_conv[0]),
							input_shape=input_shape,
							activation='relu'))

	model.add(MaxPooling3D(pool_size=(3, 3, 3)))

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

	# testando o modelo
	#model.predict(X_teste)

	model.summary()
