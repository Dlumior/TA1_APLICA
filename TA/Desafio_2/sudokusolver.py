import sys
import time
import numpy as np
from random import shuffle, random, sample, randint
from copy import deepcopy
from math import exp
import random
import matplotlib.pyplot as plt

"""gaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"""
cant_llamadas = 0 
maxScore = 0

def evaluate_population(population, fitness_fn):
    """ Evalua una poblacion de individuos con la funcion de fitness pasada """
    popsize = len(population)
    for i in range(popsize):
        if population[i].fitness == -1:    # si el individuo no esta evaluado
            population[i].fitness = fitness_fn(population[i].chromosome)



def select_parents_roulette(population):
    popsize = len(population)
    #print("imprime la longitud del popsize",popsize)
    # Escoje el primer padre
    sumfitness = sum([indiv.fitness for indiv in population])  # suma total del fitness de la poblacion
    pickfitness = random.uniform(0, sumfitness)   # escoge un numero aleatorio entre 0 y sumfitness
    cumfitness = 0     # fitness acumulado
    for i in range(popsize):
        cumfitness += population[i].fitness
        if cumfitness < pickfitness:#esta bien
            iParent1 = i
            break
            #print("paso el break del primero",i)

    # Escoje el segundo padre, desconsiderando el primer padre
    sumfitness = sumfitness - population[iParent1].fitness # retira el fitness del padre ya escogido
    pickfitness = random.uniform(0, sumfitness)   # escoge un numero aleatorio entre 0 y sumfitness
    cumfitness = 0     # fitness acumulado
    for i in range(popsize):
        if i == iParent1: continue   # si es el primer padre
        cumfitness += population[i].fitness
        if cumfitness < pickfitness:
            #print("encontrado")
            iParent2 = i
            #print("paso el break 2",i)
            break
    return (population[iParent1], population[iParent2])

def select_survivors(population, offspring_population, numsurvivors):
    next_population = []
    population.extend(offspring_population) # une las dos poblaciones
    isurvivors = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=True)[:numsurvivors]
    for i in range(numsurvivors): next_population.append(population[isurvivors[i]])
    return next_population
def select_survivors2(population, offspring_population, numsurvivors):
    next_population = []
    population.extend(offspring_population) # une las dos poblaciones
    isurvivors = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=False)[:numsurvivors]
    for i in range(numsurvivors): next_population.append(population[isurvivors[i]])
    return next_population

"""gaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"""
def get_column_indices(i, type="data index"):
	if type=="data index":
		column=1%9
	elif type=="column index":
		column = i
	indices = [column + 9 * j for j in range(9)]
	return indices

def get_row_indices(i, type="data index"):
    if type=="data index":
        row = i // 9
    elif type=="row index":
        row = i
    indices = [j + 9*row for j in range(9)]
    return indices
#################################################################################################33	LENA LOS VALORES A LOS TABLEROS (datos iniciales)
def get_block_indices(k,initialEntries,ignore_originals=False):
	row_offset = (k//3)*3
	col_offset= (k%3)*3
	indices=[col_offset+(j%3)+9*(row_offset+(j//3)) for j in range(9)]
	if ignore_originals:
		#indices = filter(lambda x:x not in initialEntries, indices)
		indices = [x for x in indices if x not in initialEntries]
	return indices

def randomAssign(puzzle, initialEntries):
	#asigna valores aleatorios a los tableros
	for num in range(9):
		block_indices=get_block_indices(num, initialEntries)
		block= puzzle[block_indices]
		zero_indices=[ind for i,ind in enumerate(block_indices) if block[i] == 0]
		to_fill = [i for i in range(1,10) if i not in block]
		shuffle(to_fill)
		for ind, value in zip(zero_indices, to_fill):
			puzzle[ind]=value
####################################################################################################################
def score_board(puzzle):#calcula el fitness
	global cant_llamadas
	global maxScore
	score = 0
	for row in range(9): # por cada fila obtiene la cantidad de numeros diferentes
		score-= len(set(puzzle[get_row_indices(row, type="row index")]))
	for col in range(9): # por cada columna obtiene la cantidad de numeros diferentes
		score -= len(set(puzzle[get_column_indices(col,type="column index")]))
	if score < maxScore :
		maxScore = score
		cant_llamadas += 1

	return score

def make_neighborBoard(puzzle, initialEntries):
    new_data = deepcopy(puzzle)
    block = randint(0,8)  # escoje un bloque aleatoriamente
    num_in_block = len(get_block_indices(block,initialEntries,ignore_originals=True)) #cantidad de posiciones que se puede mover en el bloque
    random_squares = sample(range(num_in_block),2)
    square1, square2 = [get_block_indices(block,initialEntries,ignore_originals=True)[ind] for ind in random_squares]
    new_data[square1], new_data[square2] = new_data[square2], new_data[square1]
    return new_data
def showPuzzle(puzzle):##imprime el tablero
	def checkZero(s):
		if s != 0: return str(s)
		if s == 0: return "0"
	results = np.array([puzzle[get_row_indices(j, type="row index")] for j in range(9)])
	s=""
	for i, row in enumerate(results):
		if i%3==0:
			s +="-"*25+'\n'
		s += "| " + " | ".join([" ".join(checkZero(s) for s in list(row)[3*(k-1):3*k]) for k in range(1,4)]) + " |\n"
	s +="-"*25+''
	print (s)

def sa_solver(puzzle, strParameters):
	""" Simulating annealing solver.
		puzzle: is a np array of 81 elements. The first 9 are the first row of the puzzle, the next 9 are the second row ...
		strParameters: a string of comma separated parameter=value pairs. Parameters can be:
				T0: Initial temperatura
				DR: The decay rate of the schedule function: Ti = T0*(DR)^i (Ti is the temperature at iteration i). For efficiecy it is calculated as Ti = T(i-1)*DR
				maxIter: The maximum number of iterations
	"""
	import shlex
	parameters = {'T0': .5,	'DR': .99999, 'maxIter': 100000} # Dictionary of parameters with default values
	parms_passed = dict(token.split('=') for token in shlex.split(strParameters.replace(',',' '))) # get the parameters from the parameter string into a dictionary
	parameters.update(parms_passed)  # Update  parameters with the passed values

	start_time = time.time()
	print ('Simulated Annealing intentará resolver el siguiente puzzle: ')
	showPuzzle(puzzle)

	initialEntries = np.arange(81)[puzzle > 0]  # las posiciones no vacias del puzzle
	randomAssign(puzzle, initialEntries)  # En cada box del puzzle asigna numeros aleatorios en pociciones vacias, garantizando que sean los 9 numeros diferentes
	best_puzzle = deepcopy(puzzle)
	current_score = (puzzle)
	best_score = current_score
	T = float(parameters['T0'])  # El valor inicial de la temperatura
	DR = float(parameters['DR']) # El factor de decaimiento de la temperatura
	maxIter = int(parameters['maxIter']) # El maximo numero de iteraciones
	t = 0
	#--------------------------------------------------------------------------------------
	while (t < maxIter):
		try:
			if (t % 10000 == 0):
				print('Iteration {},\tTemperaure = {},\tBest score = {},\tCurrent score = {}'.format(t, T, best_score, current_score))
			neighborBoard = make_neighborBoard(puzzle, initialEntries)
			neighborBoardScore = score_board(neighborBoard)
			delta = float(current_score - neighborBoardScore)
			if (exp((delta/T)) - random() > 0):
				puzzle = neighborBoard
				current_score = neighborBoardScore
			if (current_score < best_score):
				best_puzzle = deepcopy(puzzle)
				best_score = score_board(best_puzzle)
			if neighborBoardScore == -162:   # -162 es el score optimo
				puzzle = neighborBoard
				break
			T = DR*T
			t += 1
		except:
			print("Numerical error occurred. It's a random algorithm so try again.")
            #-----------------------------------------------------------------------------
	end_time = time.time()
	if best_score == -162:
		print ("Solution:")
		showPuzzle(puzzle)
		print ("It took {} seconds to solve this puzzle.".format(end_time - start_time))
	else:
		print("Couldn't solve! ({}/{} points). It's a random algorithm so try again.".format(best_score,-162))
#----------------------------------------------------------------------------------------------
def get_block_indices2(k):
	row_offset = (k//3)*3
	col_offset= (k%3)*3
	indices=[col_offset+(j%3)+9*(row_offset+(j//3)) for j in range(9)]
	return indices

class Individual:
	"Clase abstracta para individuos de un algoritmo evolutivo."

	def __init__(self, chromosome):
		self.chromosome = chromosome
class Individual_sudoku(Individual):
	"Clase que implementa el individuo en el problema de las n-reinas."

	def __init__(self, chromosome):
		self.chromosome = chromosome[:]
		self.fitness = -1
	###############################POR VERIFICAR
	def crossover_onepoint(self, other):
		"Retorna dos nuevos individuos del cruzamiento de un punto entre self y other "
		#implementacio single o cruzamiento de un punto
		c = random.randrange(81)
		pos=c//9 #se hace division entera
		chromosome1 = deepcopy(self.chromosome)
		chromosome2 = deepcopy(other.chromosome)

		""" ESto se debe devolver a lo normal
		print("imprime la posicion",pos)
		print("imprime lo heredado del primero",self.chromosome[:pos])
		chromosome1=self.chromosome[:pos]+other.chromosome[pos:]
		print("el cromosoma1",chromosome1)
		chromosome2=other.chromosome[:pos]+self.chromosome[pos:]
		ind1 = Individual_sudoku(chromosome1)
		ind2 = Individual_sudoku(chromosome2)
		#################
		"""
		for i in range(pos):
			block_indices=get_block_indices2(i)##obtiene la lista para cambiar los bloques
			chromosome1[block_indices]=other.chromosome[block_indices]#le pasa al cromosome 1 del other todos los bloques hasta pos
			chromosome2[block_indices] =self.chromosome[block_indices]#del self se le pasa los bloques hasta el pos

		ind1 = Individual_sudoku(chromosome1)
		ind2 = Individual_sudoku(chromosome2)
		return [ind1, ind2]


	def crossover_uniform(self, other):
		chromosome1 = deepcopy(self.chromosome)
		chromosome2 = deepcopy(other.chromosome)
		"Retorna dos nuevos individuos del cruzamiento uniforme entre self y other "
		#Cambiamos la forma de cruzamiento debido a que cada chromosome es un conjunto de 9 valores
		#IMPLEMENTACION Nueva
		for i in range(0,9): #Numero de bloques  #ESTO LO HE MODIFICADO
			if random.uniform(0, 1) < 0.5: #Cambiamos los bloques
				block_indices=get_block_indices2(i)
				chromosome2[block_indices]=self.chromosome[block_indices]
				chromosome1[block_indices] = other.chromosome[block_indices]
			#else: Ya esta cambiado
		ind1 = Individual_sudoku(chromosome1)
		ind2 = Individual_sudoku(chromosome2)
		return [ind1, ind2]
	def mutate_position(self,puzzle,initialEntries):#tiene  que recibir como parametro el puzzle inicial osea la plantilla
		"Cambia aleatoriamente la posicion de los alelos de un tablero."
		#crea un individuo que es el mutated_ind

		pos=randint(0,8)#Ya que solo un alelo se debe modificar segun los datos

		mutated_ind = deepcopy(self.chromosome)
		auxiliar = deepcopy(puzzle)
		randomAssign(auxiliar, initialEntries)#crea una nuevo individuo (lo inicia con datos aleatorios)
		#se creo un poblador pero se le transfiere un alelo
		block_indices=get_block_indices2(pos);

		mutated_ind[block_indices]=auxiliar[block_indices]

		individuo_mutado=Individual_sudoku(mutated_ind)
		individuo_mutado.fitness = score_board(individuo_mutado.chromosome)
		return individuo_mutado
   #####################################################POR VERIFICAR
def ga_solver(puzzle, strParameters):
	""" Genetic Algorithm solver.
		puzzle: is a np array of 81 elements. The first 9 are the first row of the puzzle, the next 9 are the second row ...
		strParameters: a string of comma separated parameter=value pairs. Parameters can be:
				w: Population size
				Cx: Crossover ( single  or uniform )
				m: Mutation rate
				maxGener: The maximum number of generations
	"""
	import shlex
	parameters = {'w': 10,	'Cx': 'single', 'm': 0.1,'maxGener':10000} # Dictionary of parameters with default values
	parms_passed = dict(token.split('=') for token in shlex.split(strParameters.replace(',',' '))) # get the parameters from the parameter string into a dictionary
	parameters.update(parms_passed)  # Update  parameters with the passed values

	start_time = time.time()
	print ('Genetic algorithm intentará resolver el siguiente puzzle: ')
	showPuzzle(puzzle)
	#esto esta siendo editado
	initialEntries = np.arange(81)[puzzle > 0]  # las posiciones no vacias del puzzle

	w = int(parameters['w'])  # El valor inicial de la poblacion
	Cx = str(parameters['Cx']) # El tipo  de  crossover(single or uniform)
	m=float(parameters['m']) #almacena el mutate rate
	maxGener = int(parameters['maxGener']) # El maximo numero de generaciones

	population=[]
	for i in range(w):#el w es el popsize (el tamaño de la poblacion inicial)
		new_data = deepcopy(puzzle)
		randomAssign(new_data, initialEntries)
		population.append(Individual_sudoku(new_data))#population almacena los individuos iniciales

	popsize=len(population)#el tamaño de los individuos
	evaluate_population(population,score_board) #Actualiza el fitness de cada individuo

	print("imprime los individuos iniciales:")
	for i in population:
		showPuzzle(i.chromosome)
		print("fitness: ",i.fitness)
	ibest = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=False)[:1]
	bestfitness = [population[ibest[0]].fitness]
	print("Poblacion inicial, best_fitness = {}".format(population[ibest[0]].fitness))

	#dentro de la parte def genetic_algorithm
	#------------------------------------------------------------
	fitnessAnterior = 0
	for g in range(maxGener):   # Por cada generacion

		## Selecciona las parejas de padres para cruzamiento
		mating_pool = []
		for i in range(int(popsize/2)):
			mating_pool.append(select_parents_roulette(population))
		## Crea la poblacion descendencia cruzando las parejas del mating pool con Recombinación de 1 punto
		offspring_population = []
		for i in range(len(mating_pool)):
			#offspring_population.extend( mating_pool[i][0].crossover_onepoint(mating_pool[i][1])
			if Cx == 'single':
				offspring_population.extend(mating_pool[i][0].crossover_onepoint(mating_pool[i][1]) )
			elif Cx == 'uniform':
				offspring_population.extend(mating_pool[i][0].crossover_uniform(mating_pool[i][1]) )
			################offspring_population.extend(mating_pool[i][0].crossover_uniform(mating_pool[i][1]) )
		#print("ERROR1")
		for i in range(len(offspring_population)):
			if random.uniform(0, 1) < m:# se compara con el factor de mutacion
				#print("ERROR 12")
				offspring_population[i] = offspring_population[i].mutate_position(puzzle,initialEntries)#------------------------------------------------
				#print("ERROR 13")
		evaluate_population(offspring_population, score_board)  # evalua la poblacion inicial
		#print("ERROR2")
        ## Selecciona popsize individuos para la sgte. generación de la union de la pob. actual y  pob. descendencia
		population = select_survivors2(population, offspring_population, popsize)
	#------------------------------------------------------------
		## Almacena la historia del fitness del mejor individuo
		ibest = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=False)[:1]
		bestfitness.append(population[ibest[0]].fitness)
		if population[ibest[0]].fitness<fitnessAnterior:
			print("generacion {}, best_fitness = {}, cantidad de llamadas a score_board= {}".format(g, population[ibest[0]].fitness,cant_llamadas))
			fitnessAnterior = population[ibest[0]].fitness

	end_time = time.time()
	if population[ibest[0]].fitness == -162:
		print ("========= SOLUCION ===========")
		showPuzzle(population[ibest[0]].chromosome)
		print("Fitness = {}".format(population[ibest[0]].fitness))
		print ("It took {} seconds to solve this puzzle.".format(end_time - start_time))
	else:
		print("======== SUDOKU FINAL =========")
		showPuzzle(population[ibest[0]].chromosome)
		print("Fitness = {}".format(population[ibest[0]].fitness))
		print ("It took {} seconds".format(end_time - start_time))
		print("Couldn't solve! ({}/{} points). It's a random algorithm so try again.".format(population[ibest[0]].fitness,-162))


##-------------------------------------------------------------------------------------------------------------------
def default(str):
    return str + ' [Default: %default]'

def readCommand( argv ):
	"""
	Processes the arguments  used to run sudokusolver from the command line.
	"""
	from optparse import OptionParser
	usageStr = """
	USAGE:      python sudokusolver.py <options>
	EXAMPLES:   (1) python sudokusolver.py -p my_puzzle.txt -s sa -a T0=0.5,DR=0.9999,maxIter=100000
	"""
	parser = OptionParser(usageStr)
	parser.add_option('-p', '--puzzle', dest='puzzle', help=default('the puzzle filename'), default=None)
	parser.add_option('-s', '--solver', dest='solver', help=default('name of the solver (sa or ga)'), default='sa')
	parser.add_option('-a', '--solverParams', dest='solverParams', help=default('Comma separated pairs parameter=value to the solver. e.g. (for sa): "T0=0.5,DR=0.9999,nIter=100000"'))

	options, otherjunk = parser.parse_args(argv)
	if len(otherjunk) != 0:
		raise Exception('Command line input not understood: ' + str(otherjunk))
	args = dict()

	fd = open(options.puzzle,"r+")    # Read the Puzzle file
	puzzle = eval(fd.readline())
	array = []
	for row in puzzle:
		for col in row:
			array.append(col)
	args['puzzle'] = np.array(array)  # puzzle es un vector con todas las filas del puzzle concatenadas (vacios tiene valor 0)
	args['solver'] = options.solver
	args['solverParams'] =  options.solverParams
	return args

if __name__=="__main__":


	"""
	The main function called when sudokusolver.py is run from the command line:
	> python sudokusolver.py

	See the usage string for more details.

	> python sudokusolver.py --help

    """

	args = readCommand( sys.argv[1:] ) # Get the arguments from the command line input
	solvers = {'sa': sa_solver,	'ga': ga_solver }  # Dictionary of available solvers

	solvers[args['solver']]( args['puzzle'], args['solverParams'] )  # Call the solver method passing the string of parameters

	pass
