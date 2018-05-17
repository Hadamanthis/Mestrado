import cv2
import numpy as np
from math import floor

import numpy as np
import cv2
from math import ceil, floor

# Falar dos resultados das divisões circular e elíptica mostrando se a adição de mais informação ajuda ou não da classificação

def cartesian_cut(img, factor):
	imgs = []
	
	fx = floor(img.shape[0]/factor)
	fy = floor(img.shape[1]/factor)
	
	for i in list(range(factor)):
		hx1, hx2 = i*fx, (i+1)*fx
		for j in list(range(factor)):
			hy1, hy2 = j*fy, (j+1)*fy
			imgi = img[hx1:hx2, hy1:hy2]
			imgs.append(imgi)
		
	return imgs

def diagonal_cut(img):
	rows = img.shape[0]
	cols = img.shape[1]

	halfR = floor(rows/2)
	halfC = floor(cols/2)

	a = rows/cols

	# Para a primeira e segunda metade de um corte vertical, respectivamente
	cut1 = img[0:halfR, 0:cols+1].copy()
	cut2 = img[halfR:rows+1, 0:cols+1].copy()
	cut3 = img[0:rows+1, 0:halfC].copy()
	cut4 = img[0:rows+1, halfC:cols+1].copy()

	for i in list(range(halfC+1)):
		y = floor(a*i)
		#print(y)
		for j in list(range(y, halfR)):
			cut1[j, i] = 0
			cut1[j, cols - i - 1] = 0
			cut2[halfR - j - 1, i] = 0
			cut2[halfR - j - 1, cols - i - 1] = 0
	
	a = cols/rows

	for i in list(range(halfR+1)):
		y = floor(a*i)
		for j in list(range(y, halfC)):
			cut3[i, j] = 0
			cut3[rows - i - 1, j] = 0
			cut4[i, halfC - j - 1] = 0
			cut4[rows - i - 1, halfC - j - 1] = 0 

	results = []

	results.append(cut1)
	results.append(cut2)
	results.append(cut3)
	results.append(cut4)

	return results

def complete_cut(img):
	imgs = diagonal_cut(img)

	results = []

	first = imgs[0]
	second = imgs[1]
	third = imgs[2]
	fourth = imgs[3]

	results.append(first[:, 0:int(first.shape[1]/2)])
	results.append(first[:, int(first.shape[1]/2):first.shape[1]])
	results.append(second[:, 0:int(second.shape[1]/2)])
	results.append(second[:, int(second.shape[1]/2):second.shape[1]])
	results.append(third[0:int(third.shape[0]/2), :])
	results.append(third[int(third.shape[0]/2):third.shape[0], :])
	results.append(fourth[0:int(fourth.shape[0]/2), :])
	results.append(fourth[int(fourth.shape[0]/2):fourth.shape[0], :])

	return results

def circular_cut(img, factor, rings=False):

	# Cortando a imagem baseado no tamanho do circulo
	# Se o número de linhas é maior que o número de colunas
	if (img.shape[0] > img.shape[1]):
		h1 = img.shape[0]
		h2 = img.shape[1]

		img = img[round((h1-h2)/2):round(h2 + (h1-h2)/2), :]	
	
	# Se o número de colunas é maior que o número de linhas
	else:
		w1 = img.shape[1]
		w2 = img.shape[0]

		img = img[:, round((w1-w2)/2):round(w2 + (w1-w2)/2)]

	center = (int(img.shape[1]/2), int(img.shape[0]/2))
	#print('center = (%s, %s)' % (center))

	results = []

	for i in list(range(factor)):
		results.append(np.zeros((img.shape[0], img.shape[1]), np.uint8))
	
	# raio ponderado (arredondado para baixo)
	r = floor(img.shape[0]/(2*factor))

	#print('radius:', r)
	
	print(img.shape)

	# Percorrendo a imagem quadrada
	for x in list(range(img.shape[0])):
		for y in list(range(img.shape[1])):
			
			# distancia do ponto (x, y) até o centro
			d = pow(pow(x - center[0],2) + pow(y - center[1],2), 0.5)
			
			if (d > factor*r):
				continue

			index = ceil(d/r)		

			if (not rings):
				for c in list(range(factor, index-1, -1)):
					results[factor - 1 - c][x][y] = img[x][y]
			else:
				results[index-1][x][y] = img[x][y]
			
	return results

def eliptical_cut(img, factor, rings=False):
	results = []
	
	for i in list(range(factor)):
		results.append(np.zeros((img.shape[0], img.shape[1]), np.uint8))
	
	if (img.shape[0] > img.shape[1]):
		B = img.shape[1]
		S = img.shape[0]
	else:
		B = img.shape[0]
		S = img.shape[1]
	
	Bmin = B/(2*factor)
	Smin = S/(2*factor)
	
	center = (img.shape[0]/2, img.shape[1]/2)
	
	for x in list(range(img.shape[0])):
		for y in list(range(img.shape[1])):
			soVai = False
			for f in list(range(1, factor+1)):
		
				Sc = Smin*f
				Bc = Bmin*f
			
				out = pow((x - center[0])/Bc , 2) + pow((y - center[1])/Sc, 2)
				
				if (out <= 1):
					soVai = True			
			
				if (soVai):
					if (rings):
						results[f-1][x][y] = img[x][y]
						break
					else:
						for g in list(range(f-1, factor)):
							results[g][x][y] = img[x][y]
						break
			

	return results
	
if __name__ == '__main__':

	#img = np.array([[5, 4, 3, 2, 3, 7, 2, 4, 7], [6, 7, 7, 2, 4, 7, 8, 2, 1], [3, 5, 10, 8, 7, 2, 4, 7, 2], [6, 7, 2, 4, 7, 6, 8, 9, 3], [7, 2, 4, 7, 7, 2, 4, 7, 11], [7, 2, 4, 7, 11, 7, 2, 4, 7], [7, 2, 7, 2, 4, 7, 4, 7, 11]])
	
	img = cv2.imread('/home/geovane/Imagens/teste.png', 0)
	cv2.imshow('img', img)
	print(img.shape)
	
	imgs = eliptical_cut(img, 3)
	
	for i in list(range(len(imgs))):
		cv2.imshow('img'+str(i), imgs[i])
		
	cv2.waitKey(0)
