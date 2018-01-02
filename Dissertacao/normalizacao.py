# Dado um vetor com uma quantidade R de imagens, quero amostrar D imagens de forma que o vetor fique com amostras temporais igualmente espaçadas
import math

def normalizar(V, R, D, verbose = False):

	if (R > D):
		while (len(V) > D):
			# multiplos a serem retirados
			n = math.ceil(len(V)/abs(len(V) - D))
			
			if (verbose): 
				print("n:", n)

			L = len(V)
	
			for i in range(L, 0, -1):
				if (i%n == 0):
					V.pop(i-1)
			
			if (verbose):
				print(V)
	else:
		while (len(V) < D):
			# multiplos a serem duplicados
			n = math.ceil(len(V)/abs(len(V) - D))
			
			if (verbose):
				print("n:", n)

			L = len(V)

			for i in range(L, 0, -1):
				if (i%n == 0):
					V.insert(i-1, V[i-1])

			if (verbose):
				print(V)

	return V

def main():
	R = 5
	D = 17

	V = [i for i in range(1, R+1)]

	print("V:", V)

	V = normalizar(V, R, D)
	
	print("V:", V)
 

if __name__ == '__main__':
	main()
