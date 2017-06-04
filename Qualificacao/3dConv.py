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
folder = '/mnt/DADOS/LISA_HG_Data/dynamic_gestures/data'
n_classes = 32
n_epochs = 50

if __name__ == "__main__":
	
	# Mudar ordem de linhas e colunas
	K.set_image_dim_ordering('th')

	if K.image_dim_ordering() == 'th':
		print('Usando a ordenação do Theano')
	else:
		print('Usando a ordenacao do TensorFlow')
	
	#############################################################
	### Ler imagens (250x115) e coloca-las em uma variável X. ###
	### Há uma qtd variavel de frames por vídeo				  ###	
	### Tem 16 usuarios, performando 32 gestos 3 vezes cada   ###
	#############################################################
	
	# shape do video
	vid_shape = {'rows':50, 'cols':100, 'channels':6} 
	n_frames= 6
	
	# Dados de treinamento
	X = [] # variavel pra guardar todas as entradas
	labels = [] # Labels das classes

	# Lendo o dataset
	X, labels = readDataset(folder, n_frames)

	# Transformando entrada e labels em array Numpy
	X = np.array(X)
	Y = np.array(labels)
	num_samples = X.shape[0]
	
	print('X.shape:', X.shape)
	print('Y.shape:', Y.shape)

	# input_shape(qtd_imagens, qtd_canais, qtd_linhas, qtd_colunas, qtd_profundidade)
	input_shape = (1, vid_shape['rows'], vid_shape['cols'], vid_shape['channels'])
	train_set = np.zeros((num_samples, 1, vid_shape['rows'], vid_shape['cols'], vid_shape['channels']))

	for h in list(range(num_samples)):
		train_set[h][0][:][:][:] = X[h,:,:,:]

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
																	test_size = 0.4,
																	random_state = 4)

	print('X_treino:', X_treino.shape)
	print('X_teste:', X_teste.shape)
	print('Y_treino:', Y_treino.shape)
	print('Y_teste:', Y_teste.shape)
	
	'''
	###################################################
	### (Hardwired)									###
	### Transformar cada uma das imagens            ###
	### em 5 (nivel de cinza,   			        ###
	###		gradiente_x, gradiente_y,				###
	###		optflow_x, optflow_y) 		 			###
	###################################################
	
	treino_gray = []
	treino_grdx = []
	treino_grdy = []
	#treino_optflwx = []
	#treino_optflwy = []

	for img in X_treino:
		treino_gray.append( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ) # niveis de cinza
		treino_grdx.append( cv2.cvtColor(np.absolute(cv2.Sobel(img,cv2.CV_16U,1,0,ksize=5)), cv2.COLOR_BGR2GRAY) ) # gradiente x
		treino_grdy.append( cv2.cvtColor(np.absolute(cv2.Sobel(img,cv2.CV_16U,0,1,ksize=5)), cv2.COLOR_BGR2GRAY) ) # gradiente y
		#treino_optflwx.append(  )
		#treino_optflwy.append(  )
	
	X_tr = []
	X_tr += treino_gray
	#X_tr += treino_grdx
	#X_tr += treino_grdy

	X_tr = np.array(X_tr)

	teste_gray = []
	teste_grdx = []
	teste_grdy = []
	#teste_optflwx = []
	#teste_optflwy = []

	for img in X_teste:
		teste_gray.append( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ) # niveis de cinza
		teste_grdx.append( cv2.cvtColor(np.absolute(cv2.Sobel(img,cv2.CV_16U,1,0,ksize=5)), cv2.COLOR_BGR2GRAY) ) # gradiente x
		teste_grdy.append( cv2.cvtColor(np.absolute(cv2.Sobel(img,cv2.CV_16U,0,1,ksize=5)), cv2.COLOR_BGR2GRAY) ) # gradiente y
		#teste_optflwx.append(  )
		#teste_optflwy.append(  )
	
	X_ts = []
	X_ts += teste_gray
	X_ts += teste_grdx
	X_ts += teste_grdy

	X_ts = np.array(X_ts)

	print('X_tr.shape', X_tr.shape)
	print(Y_treino)
	print('X_ts.shape', X_ts.shape)
	print(Y_teste)
	'''

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
	n_filters = [32, 32]

	# Profundidade de convolução utilizado em cada layer (CONV x CONV)
	n_conv = [3, 3, 3]
	n_width = [30, 79, 30]
	n_height = [20, 35, 12]

	# Criar modelo
	model = Sequential()

	#model.add(Convolution3D(n_filters[0],
	#						(n_height[0], n_width[0], n_conv[0]),
	#						input_shape=input_shape,
	#						activation='relu'))

	#model.add(Dropout(0.5))

	#model.add(Convolution3D(n_filters[1], 
	#						(n_width[1], n_height[1], n_conv[1]),
	#		activation='relu'))

	#model.add(Dropout(0.5))
	
	#model.add(Convolution3D(n_filters[1], 
	#					(n_width[1], n_height[1], n_conv[1]),
	#		activation='relu'))

	#model.add(Flatten())

	#model.add(Dense(n_classes, kernel_initializer='normal'))

	#model.add(Activation('softmax'))

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

	batch_size = 2
	
	# Treinando o modelo
	hist = model.fit(X_treino, Y_treino,  validation_data=(X_teste, Y_teste), epochs=n_epochs, shuffle=True, verbose=1)

	# Testando o modelo
	score = model.evaluate(X_teste, Y_teste)
	print('\nTest score:', score[0])
	print('Test accuracy', score[1])

	model.summary()
