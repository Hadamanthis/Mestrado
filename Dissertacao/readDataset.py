import os
import cv2
import numpy as np
import normalizacao as norm

def readDataset(folder, n_frames, color = cv2.COLOR_BGR2GRAY, resizeShape = None, interpolation = cv2.INTER_AREA):
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

	X = [] # variavel pra guardar todas as entradas
	labels = [] # Labels das classes

	# Listando os videos no diretório
	classes = os.listdir(folder)

	c = 0

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
				while (cap.isOpened()):
					
					grabbed, frame = cap.read() # Lendo um frame

					if (grabbed == False):
						break
					
					if resizeShape != None:
						frame = cv2.resize(frame, (resizeShape[1], resizeShape[0]), interpolation=interpolation)
					
					if color != None:
						frame = cv2.cvtColor(frame, color)
					
					frames.append(frame)

				cap.release() # Fechando o video

				# Adicionando individuo e label
				print('frames length:', len(frames))
				frames = norm.normalizar(frames, len(frames), n_frames)
				
				print('frames length after normalization:', len(frames))
				frames = np.array(frames)
				frames = np.rollaxis(np.rollaxis(frames, 2, 0), 2, 0)
				X.append(frames)
				labels.append(c)
				
			c += 1
	
	return X, labels 
