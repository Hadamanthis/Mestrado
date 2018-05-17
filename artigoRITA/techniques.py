import cv2
import numpy as np
from convolution import conv2D

def s(x, T):
	if (x > T):
		return 1
	else:
		return 0

''' Uma função que aplica LBP 3x3 '''
def LBP(img):
	# Kernel de pesos padrão 3x3
	kernel = np.array([[1, 2, 4], [8, 0, 16], [32, 64, 128]])
	
	outputs = []
	
	# Percorrendo a imagem
	for i in list(range(1, img.shape[0]-1)):
		for j in list(range(1, img.shape[1]-1)):
			
			subimg = []
			
			for g in list(range(-1, 2)):
				# Copiando a linha
				row = []
				for k in list(range(-1, 2)):
					
					if ((g == 0) and (k == 0)):
						row.append(0)
						continue
				
					thresh = int(img[i+g][j+k]) - int(img[i][j])
					if (thresh >= 0):
						row.append(1)
					else:
						row.append(0)
				
				# Adicionando a linha
				subimg.append(np.array(row))
				
			subimg = np.array(subimg)
			outputs.append(conv2D(subimg, kernel))
			#print(subimg)
			
	#print(outputs)
	return np.array(outputs)

''' Uma função que aplica o CS-LBP com threshold T '''
def CS_LBP(img, T = None):
	
	outputs = []
	
	if (T == None):
		T = 2.56
	
	# Percorrendo a imagem
	for i in list(range(1, img.shape[0]-1)):
		for j in list(range(1, img.shape[1]-1)):
		
			out1 = img[i][j+1] - img[i][j-1]
			out2 = img[i+1][j+1] - img[i-1][j-1]
			out3 = img[i+1][j] - img[i-1][j]
			out4 = img[i+1][j-1] - img[i-1][j+1]
		
			output = s(out1, T)*pow(2,0) + s(out2, T)*pow(2, 1) + s(out3, T)*pow(2, 2) + s(out4, T)*pow(2,3)
			outputs.append(output)
		
	return np.array(outputs)
	
def LQP(img, T1, T2):
	# Kernel de pesos padrão 3x3
	kernel = np.array([[1, 2, 4], [8, 0, 16], [32, 64, 128]])
	
	outputs1 = []
	outputs2 = []
	outputs3 = []
	outputs4 = []	
	
	# Percorrendo a imagem
	for i in list(range(1, img.shape[0]-1)):
		for j in list(range(1, img.shape[1]-1)):
			subimg1 = []
			subimg2 = []
			subimg3 = []
			subimg4 = []
			
			for g in list(range(-1, 2)):
				# Copiando a linha
				row1, row2, row3, row4 = [], [], [], [] # Linhas respectivas de cada uma das subimagens
				
				for k in list(range(-1, 2)):
					
					# Se for o centro, coloca 0 e passa
					if ((g == 0) and (k == 0)):
						row1.append(0)
						row2.append(0)
						row3.append(0)
						row4.append(0)
						continue
				
					diff = img[i+g][j+k] - img[i][j]
					if (diff >= T2):
						row1.append(1)
						row2.append(0)
						row3.append(0)
						row4.append(0)
					elif (diff < T2 and diff >= T1):
						row1.append(0)
						row2.append(1)
						row3.append(0)
						row4.append(0)
					elif (diff < T1 and diff >= -1*T1):
						row1.append(0)
						row2.append(0)
						row3.append(1)
						row4.append(0)
					else:
						row1.append(0)
						row2.append(0)
						row3.append(0)
						row4.append(1)
				
				# Adicionando a linha
				subimg1.append(np.array(row1))
				subimg2.append(np.array(row2))
				subimg3.append(np.array(row3))
				subimg4.append(np.array(row4))
				
			subimg1 = np.array(subimg1)
			subimg2 = np.array(subimg2)
			subimg3 = np.array(subimg3)
			subimg4 = np.array(subimg4)
			
			outputs1.append(conv2D(subimg1, kernel))
			outputs2.append(conv2D(subimg2, kernel))
			outputs3.append(conv2D(subimg3, kernel))
			outputs4.append(conv2D(subimg4, kernel))
		
	return np.array(outputs1), np.array(outputs2), np.array(outputs3), np.array(outputs4)
	
def CLBP(img):
	# Kernel de pesos padrão 3x3
	kernel = np.array([[1, 2, 4], [8, 0, 16], [32, 64, 128]])
	
	outputs1 = []
	outputs2 = []
	
	# Percorrendo a imagem
	for i in list(range(1, img.shape[0]-1)):
		for j in list(range(1, img.shape[1]-1)):
			
			mavg = 0
			
			# Calculando a média das magnitudes da diferença entre o pixel central e os vizinhos
			for g in list(range(-1, 2)):
				for k in list(range(-1, 2)):
					
					if ((g == 0) and (k == 0)):
						continue
						
					mavg += abs(img[i][j] - img[i+g][j+k])
					
			mavg /= 8
			
			north = img[i-1][j] - img[i][j]
			northeast = img[i-1][j+1] - img[i][j]
			east = img[i][j+1] - img[i][j]
			southeast = img[i+1][j+1] - img[i][j]
			south = img[i+1][j] - img[i][j]
			southwest = img[i+1][j-1] - img[i][j]
			west = img[i][j-1] - img[i][j]
			northwest = img[i-1][j-1] - img[i][j]
			
			sum1 = 0
			sum2 = 0
			
			if (north >= 0):
				sum1 += 1
			if (abs(north) > mavg):
				sum1 += 2
			if (east >= 0):
				sum1 += 4
			if (abs(east) > mavg):
				sum1 += 8
			if (south >= 0):
				sum1 += 16
			if (abs(south) > mavg):
				sum1 += 32
			if (west >= 0):
				sum1 += 64
			if (abs(west) > mavg):
				sum1 += 128
				
			if (northeast >= 0):
				sum2 += 1
			if (abs(northeast) > mavg):
				sum2 += 2
			if (southeast >= 0):
				sum2 += 4
			if (abs(southeast) > mavg):
				sum2 += 8
			if (southwest >= 0):
				sum2 += 16
			if (abs(southwest) > mavg):
				sum2 += 32
			if (northwest >= 0):
				sum2 += 64
			if (abs(northwest) > mavg):
				sum2 += 128
			
			outputs1.append(sum1)
			outputs2.append(sum2)
				
	return np.array(outputs1), np.array(outputs2)
			
if __name__ == '__main__':
	img = np.array([[5, 4, 3, 2, 3], [6, 7, 8, 2, 1], [3, 5, 10, 8, 2], [6, 6, 8, 9, 3], [7, 2, 4, 7, 11]])
	
	print(img)
	
	outputs1, outputs2 = CLBP(img)
	
	print(outputs1)
	print(outputs2)
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
