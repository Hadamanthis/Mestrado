import cv2
import numpy as np

''' img e kernel devem ter a mesma dimensao '''
def conv2D(img, kernel):
	output = 0
	for i in list(range(img.shape[0])):
		for j in list(range(img.shape[1])):
			output += img[i, j]*kernel[i, j]
	
	return output

if __name__ == '__main__':
	img = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1]])
	kernel = np.array([[1, 2, 4], [8, 0, 16], [32, 64, 128]])

	output = conv2D(img, kernel)

	print(output)
			
