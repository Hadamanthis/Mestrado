import dicom
import os
import numpy as np
from matplotlib import pyplot, cm

def readDataset(folder):

	X = []
	label = []

	for dirName, subdirList, fileList in os.walk(folder):
		for filename in fileList:
			filename = os.path.join(dirName, filename)

			# Lendo o arquivo
			RefDs = dicom.read_file(filename)
			
			# TODO Tentar usar as imagens do jeito que foram lidas no teste 3
	
			# Se o atributo NumberOfFrames não existir então só há uma imagem
			try:						
			
				# Tem que ser maior que 2
				if (RefDs.NumberOfFrames < 3):
					continue

				print(filename)

			except:
				continue
			
			RefDs = RefDs.pixel_array
			print(RefDs.shape)
			RefDs = np.rollaxis(np.rollaxis(RefDs, 2, 0), 2, 0)
			print(RefDs.shape)
			X.append(RefDs)

			if "benigno" in filename:
				label.append(0)
			elif "maligno" in filename:
				label.append(1)
			else:
				label.append(2)

	return X, label

folder = "/home/geovane/Transferências/LIDC-IDRI"
readDataset(folder)
