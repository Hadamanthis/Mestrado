import cv2
import numpy as np

def realce_logaritmico(img):
	l = np.log10(img.astype(np.uint16)+1)
	return (l*105.9612).astype(np.uint8)
	
if __name__ == '__main__':
	img = cv2.imread('/home/geovane/Imagens/teste.png', 0)
	cv2.imshow('img', img)
	
	print(img.max(), img.min())
	
	realcada = realce_logaritmico(img)
	
	print(realcada.max(), realcada.min())
	
	cv2.imshow('realce', realcada)
	
	cv2.waitKey()
	
# PEGAR O CODIGO QUE LE AS IMAGENS, APLICAR O REALCE NELAS
# GERAR OS HISTOGRAMAS SEM ABORDAGENS DE DIVISÃO ESPACIAL COM AS NOVAS IMAGENS REALCADAS (PRA VER SE CHEGAM NA CASA DOS 80 DE F-SCORE)
# TESTAR TAMBEM FAZER O HISTOGRAMA MANUALMENTE, COMO EU FIZ DA VEZ QUE TESTEI ISSO
