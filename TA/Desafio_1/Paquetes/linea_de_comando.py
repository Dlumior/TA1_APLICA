def readCommand( argv ):
	"""
	Procesa los argumentos de la linea de comandos
	"""
	from optparse import OptionParser

	usageStr="""
	USAGE:      python findroute.py <options>
	EXAMPLES:   python findroute.py -e Archivos\edges_pitts.txt –s 275754986 -g 105012740 –m astar -l Archivos\heuristics_pitts.txt
				python findroute.py --help
	"""

	parser = OptionParser(usageStr)
	parser.add_option('-e', '--calles', dest='calles', help= 'El nombre del archivo con las calles', default='Archivos\edges_pitts.txt')
	parser.add_option('-s', '--nodoI', dest='nodoI', help= 'nodeID inicial', default='104878620')
	parser.add_option('-g', '--nodoF', dest='nodoF', help= 'nodeID final', default='105012740')
	parser.add_option('-m', '--busqueda', dest='busqueda', help= 'Metodo de busqueda e.g: bfs, dfs, etc.', default='bfs')
	parser.add_option('-l', '--heuristica', dest='heuristica', help= 'heuristica para Astar search', default="")

	options, otherjunk = parser.parse_args(argv)

	if len(otherjunk) != 0:
		raise Exception('Opciones que no se han podido entender: ' + str(otherjunk))

	e_pits_txt = open(options.calles,"r")
	lineas = e_pits_txt.readlines()
	lista=[]
	for linea in lineas:
	    lista.append(linea.split())

	mapa = dict()
	for sublista in lista:
		if int(sublista[0]) in mapa:
			mapa[int(sublista[0])].append((int(sublista[1]),float(sublista[2])))
		else:
			mapa[int(sublista[0])]=[(int(sublista[1]),float(sublista[2]))]

		if int(sublista[1]) in mapa:
			mapa[int(sublista[1])].append((int(sublista[0]),float(sublista[2])))
		else:
			mapa[int(sublista[1])]=[(int(sublista[0]),float(sublista[2]))]

	if options.heuristica!="":
		h_pitts_txt=open(options.heuristica,"r")
		lineas= h_pitts_txt.readlines()
		h_pitts=[]
		for linea in lineas:
			h_pitts.append(linea.split())

		heuristica = dict()
		for nodo in h_pitts:
			heuristica[int(nodo[0])]=float(nodo[1])

	args = dict()
	args['mapa'] = mapa
	args['nodoI'] = options.nodoI
	args['nodoF'] =  options.nodoF
	args['busqueda'] = options.busqueda

	if options.heuristica!="":
		args['heuristica'] = heuristica
	else:
		args['heuristica'] = options.heuristica

	return args
