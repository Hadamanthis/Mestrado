import os
import numpy as np
import cv2

BENIGNO = 'benignos'
MALIGNO = 'malignos'

def read_dataset(folder):

	print('> Loading', folder) 
	
	# Percorrendo todas as classes presentes na base
	for c in os.listdir(folder):
	
		# Label da classe
		if (c == BENIGNO):
			INDEX = 0
		elif (c == MALIGNO):
			INDEX = 1
		
		class_path = folder + '/' + c
		
		print(class_path)
		
		# Percorrendo todos os individuos presentes na classe
		for f in os.listdir(class_path):
		
			file_name = class_path + '/' + f

			print('> Reading:', file_name)
			
			sample = open(file_name)
			
			rows, cols = sample.readline().split('\t')
			
			rows, cols = int(rows), int(cols)
			
			img = np.ndarray((cols, rows))
			
			# Lendo os valores
			for x in list(range(cols)):
				for y in list(range(rows)):
					value = int(sample.readline())
					
					if (value < 0):
						img[x][y] = 0
					else:
						img[x][y] = value
			
			#cv2.imshow('img', img)	
			#cv2.imwrite('/home/geovane/teste.jpg', img)
			#cv2.waitKey(0)
			yield img, INDEX
		
	
if __name__ == '__main__':
	pass
	
