import os
import cv2
import numpy as np

def readDataset(folder, n_frames):
	""" Função de leitura de dataset. O dataset deve estar no formato
	folder->class_folder->samples
	
	Exemplo:
		Lendo arquivos de uma dataset localizada em '/dataset01':
			$ readDataset('/dataset01')

	Args:
		folder (string): O diretorio raiz do dataset
		n_frames (int): A quantidade de imagens a ser extraida por video

	Retorna:
		List: Uma lista das imagens em Numpy
		List: Labels lista das classes lidas no dataset
	"""
	resizeShape = (100, 50)
	interpolation = cv2.INTER_AREA
	cvtColor = cv2.COLOR_BGR2GRAY

	X = [] # variavel pra guardar todas as entradas
	labels = [] # Labels das classes

	# Listando os videos no diretório
	classes = os.listdir(folder)

	# Para cada arquivo no diretório...
	for file_name in classes:
	
		label = file_name		

		file_name = folder + '/' + file_name # Construir caminho completo
		
		# Se é um diretorio
		if (os.path.isdir(file_name)):

			print(file_name)

			listing = os.listdir(file_name)
			
			# Para cada video dentro do diretorio
			for vid in listing:

				vid = file_name + '/' + vid
				
				frames = []
				
				print(vid)

				cap = cv2.VideoCapture(vid) # Abrindo o vídeo
	
				# Testando se o vídeo esta aberto
				if (not cap.isOpened()):
					print('[ERRO] Nao foi possivel abrir', vid)	
					break

				# Para cada frame do vídeo
				for i in list(range(n_frames)):
					
					grabbed, frame = cap.read() # Lendo um frame

					# Testando se o frame foi lido corretamente
					if (not grabbed):
						print('[ERRO] Nao foi possivel ler o frame', i, 'de', vid)
						break

					frame = cv2.resize(frame, (resizeShape[0], resizeShape[1]), 										interpolation=interpolation)

					frame = cv2.cvtColor(frame, cvtColor)
					
					frames.append(frame)

				cap.release() # Fechando o video

				# Adicionando individuo e label
				frames = np.array(frames)
				frames = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
				X.append(frames)
				labels.append(int(label)-1)
	
	return X, labels 
