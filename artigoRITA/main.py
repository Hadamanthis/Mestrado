import cv2
import numpy as np
from techniques import LBP, CS_LBP, LQP, CLBP
from crop import cartesian_cut, diagonal_cut, complete_cut, circular_cut, eliptical_cut
from dataset import read_dataset

# Falta salvar os valores em um arquivo
		
			# Falar disso no artigo:
			#The number of bins used in quantization of the feature  space plays a crucial role. Histograms with too modest a number of bins fail to provide enough discriminative information about the distributions. However, since the distributions have a finite amount of entries, it does not make sense to go to the other extreme. If histograms have too many bins, and the average number of entries per bin is very small, histo- grams become sparse and unstable.
			# Nos meus testes utilizando somente o histograma, a quantidade de bins não foi tão importante assim, gerando uma diferença de resultados da ordem de 2%?!
			
			# Da normalização do histograma e de como a quantidade de bins foi decidida
			
			# The other parameters (i.e. t, t1 and t2) have been fixed for all the experiments without optimization, anyway empirical tests have shown very similar performance is for different values for t, t1 and t2. (NANNI(2010))
			
			# Apliquei um downsize com fator de redução 3 em ambas as dimensões

TECHNIQUE = 3 # 1- LBP/CS-LBP, 2 - LQP, 3 - CLBP

n_bins = [16, 32, 64, 128, 256]
folder = '/home/geovane/Transferências/base_simara_sem_melhoria'
	
imgs = read_dataset(folder)
	
for frame, label in imgs:

	frame = cv2.resize(frame, None, fx=1/3, fy=1/3)

	#subimgs = eliptical_cut(frame, 2)
	
	subimgs = [frame]
	
	for b in n_bins:
	
		if (TECHNIQUE == 1):
			
			output_file_path = '/home/geovane/CS_LBP_%sbins.txt' % (b)
			print(output_file_path)
			
			output_file = open(output_file_path, 'a')
			output_file.write('%s\t' % label)
			
			# contador de features
			g = 1
			
			for img in subimgs:
			
				# Apply LBP/CS-LBP
				result = CS_LBP(img)
				
				hist, bins = np.histogram(result, b)
				hist = hist/float(sum(hist))
				print(hist)
		
				for feature in hist:
					output_file.write('%s:%s\t' % (g, str(feature)))
					g += 1
			
			output_file.write('\n')
				
			output_file.close()
					
		elif (TECHNIQUE == 2):
		
			output_file_path = '/home/geovane/LQP_1_10_%sbins.txt' % (b)
			print(output_file_path)
			
			output_file = open(output_file_path, 'a')
			output_file.write('%s\t' % label)

			# contador de features
			g = 1			
			
			for img in subimgs:
			
				# Apply LQP
				result1, result2, result3, result4 = LQP(img, 1, 10)
				
				hist1, bins = np.histogram(result1, b)
				hist2, bins = np.histogram(result2, b)
				hist3, bins = np.histogram(result3, b)
				hist4, bins = np.histogram(result4, b)
			
				# Concatenando histogramas
				hist = hist1.tolist() + hist2.tolist() + hist3.tolist() + hist4.tolist()
				hist = np.array(hist)/float(sum(hist))
				print(hist)
		
				for feature in hist:
					output_file.write('%s:%s\t' % (str(g), str(feature)))
					g += 1
			
			output_file.write('\n')
			output_file.close()
			
		else:

			output_file_path = '/home/geovane/CLBP_%sbins.txt' % (b)
			print(output_file_path)
			
			output_file = open(output_file_path, 'a')
			output_file.write('%s\t' % label)
			
			# contador de features
			g = 1

			for img in subimgs:
			
				# Apply CLBP
				result1, result2 = CLBP(img)
				
				hist1, bins = np.histogram(result1, b)
				hist2, bins = np.histogram(result2, b)
			
				# Concatenando histogramas
				hist = hist1.tolist() + hist2.tolist()
				hist = np.array(hist)/float(sum(hist))
				print(hist)
			
				for feature in hist:
					output_file.write('%s:%s\t' % (str(g), str(feature)))
					g += 1
			
			output_file.write('\n')
			output_file.close()
