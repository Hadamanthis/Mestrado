import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

# Diretório do dataset
base_folder = '/mnt/DADOS/LISA_HG_Data/dynamic_gestures/data'
n_classes = 33
n_epochs = 50

if __name__ == "__main__":
	
	#############################################################
	### Ler imagens (250x115) e coloca-las em uma variável X. ###
	### Há uma qtd variavel de frames por vídeo				  ###	
	### Tem 16 usuarios, performando 32 gestos 3 vezes cada   ###
	#############################################################

	img_shape = (250, 115, 6) # shape do video img_rows, img_cols, qtd de frames
	
	# Dados de treinamento
	X = [] # variavel pra guardar todas as entradas
	labels = [] # Labels das classes

	# Listando os videos no diretório
	classes = os.listdir(base_folder)

	# Para cada arquivo no diretório...
	for file_name in classes:
	
		label = file_name		

		file_name = base_folder + '/' + file_name # Construir caminho completo
		
		# Se é um diretorio
		if (os.path.isdir(file_name)):

			print(file_name)

			listing = os.listdir(file_name)
			
			# Para cada video dentro do diretorio
			for vid in listing:

				vid = file_name + '/' + vid

				print(vid)

				cap = cv2.VideoCapture(vid) # Abrindo o vídeo
	
				# Testando se o vídeo esta aberto
				if (not cap.isOpened()):
					print('[ERRO] Nao foi possivel abrir', vid)	
					break
		
				# Pegando o numero de frames dos videos
				n_frames = int( cap.get(cv2.CAP_PROP_FRAME_COUNT) )	

				# Para cada frame do vídeo
				for i in list(range(img_shape[2])): # Deveria ser list(range(n_frames))
			
					grabbed, frame = cap.read() # Lendo um frame

					# Testando se o frame foi lido corretamente
					if (not grabbed):
						print('[ERRO] Nao foi possivel ler o frame', i, 'de', vid)
						break
					
					np_frame = np.array(frame)
					
					# Adicionando individuo e label
					X.append(np_frame)
					labels.append(int(label))

				cap.release() # Fechando o video
	
	#print(len(X))
	#print(len(labels))
	
	print(len(X))
	print(len(labels))

	# Transformando entrada e labels em array Numpy
	X = np.array(X)
	Y = np.array(labels)

	print(X.shape)
	print(Y.shape)

	# Separando os dados em treino/teste 80/20
	X_treino, X_teste, label_treino, label_teste = train_test_split(X, Y,
																	test_size = 0.2,
																	random_state = 4)

	# Transformando o vetor de classes em matrizes de classes binárias
	Y_treino = np_utils.to_categorical(label_treino, n_classes)
	Y_teste = np_utils.to_categorical(label_teste, n_classes)
	
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
	
	print('X_tr.shape', X_tr.shape)
	print(Y_treino)
	print('X_ts.shape', X_ts.shape)
	print(Y_teste)

	# Numero de filtros convolucionais em cada layer
	n_filters = [2, 3]

	# Profundidade de convolução utilizado em cada layer (CONV x CONV)
	n_conv = [3, 3, 3]
	n_width = [151, 79, 30]
	n_height = [70, 35, 12]

	# Criar modelo
	model = Sequential()

	model.add(Convolution3D(n_filters[0],
							(n_width[0], n_height[0], n_conv[0]),
			 input_shape=(X_tr.shape[0], img_shape[0], img_shape[1], img_shape[2]*3),
			 activation='tanh'))

	model.add(Dropout(0.5))

	model.add(Convolution3D(n_filters[1], 
							(n_width[1], n_height[1], n_conv[1]),
			activation='tanh'))

	model.add(Dropout(0.5))
	
	model.add(Convolution3D(n_filters[1], 
						(n_width[1], n_height[1], n_conv[1]),
			activation='tanh'))

	model.add(Dense(n_classes, kernel_initializer='normal'))

	model.add(Activation('tanh'))

	# Cada ação é classificada da maneira "one-against-rest"

	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	
	# Treinando o modelo
	hist = model.fit(X_tr, Y_treino,  validation_data=(X_ts, Y_teste), epochs=n_epochs, shuffle=True)

	# Testando o modelo
	score = model.evaluate(X_ts, Y_teste)
	print('\nTest score:', score[0])
	print('Test accuracy', score[1])
	
	# A performance do reconhecimento tirou a média de 5 runs aleatórias.

	# O modelo 3D CNN alcançou uma acurácia média de 90.2% comparado com 91.7% do HMAX
