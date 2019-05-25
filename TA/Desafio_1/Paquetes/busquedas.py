import sys
import bisect
import collections
import collections.abc
import heapq
import operator
import os.path
import random
import math
import functools
import numpy as np
from itertools import chain, combinations

from Paquetes.definicionProblema import *

#____________________________________________________________________________________________________________________________
from collections import deque

class FIFOQueue(deque):
    """Una cola First-In-First-Out"""
    def pop(self):
        return self.popleft()
#____________________________________________________________________________________________________________________________
def depth_first_tree_search(problem,frontier):
    """ Search the deepest nodes in the search tree first.
        Search through the successors of a problem to find a goal.
        The argument frontier should be an empty queue.
        Repeats infinitely in case of loops. [Figure 3.7]
    """

    frontier.append(Node(problem.initial))
    nodos_visitados=0
    while frontier:
        node = frontier.pop()
        nodos_visitados+=1
        if problem.goal_test(node.state):
            return node,nodos_visitados,len(node.solution())
        frontier.extend(node.expand(problem))
    return None,nodos_visitados
#____________________________________________________________________________________________________________________________
def graph_search(problem, frontier):

    frontier.append(Node(problem.initial))
    explored = set()
    nodos_visitados=1
    nodos_en_memoria=1
    while frontier:
        node = frontier.pop()
        nodos_visitados+=1
        if problem.goal_test(node.state):
            return node,nodos_visitados,len(frontier)+nodos_visitados
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if( (child.state not in explored) and
                        (child not in frontier)))
    return None,nodos_visitados,len(frontier)+nodos_visitados
#____________________________________________________________________________________________________________________________

def iterative_deeping_search(problem):
    nodos_visitados=0
    for depth in range(sys.maxsize):
        result, nodos =depth_limited_search(problem, depth)
        nodos_visitados+=nodos
        if result != 'cutoff':
            return result,nodos_visitados,len(result.solution())

def depth_limited_search(problem, limit=10):
    result, nodos_visitados= recursive_dls(Node(problem.initial), problem, limit)
    return (result,nodos_visitados)

def recursive_dls(node, problem, limit):
    nodos_visitados=0
    if problem.goal_test(node.state):
        nodos_visitados+=1
        return (node,nodos_visitados)
    elif limit==0:
        nodos_visitados+=1
        return ('cutoff',nodos_visitados)
    else:
        cutoff_occurred=False
        for child in node.expand(problem):
            result,nodos_visitadosh=recursive_dls(child,problem, limit-1)
            nodos_visitados+=nodos_visitadosh

            if result=='cutoff':
                cutoff_occurred=True

            elif result is not None:
                return (result,nodos_visitados)
        return ('cutoff',nodos_visitados) if cutoff_occurred else (None,nodos_visitados)
#____________________________________________________________________________________________________________________________

def interseccion(lista1, lista2):
    intrs1=[value for value in lista1
            if value.state in [estado.state for estado in lista2]]
    intrs2=[value for value in lista2
            if value.state in [estado.state for estado in lista1]]
    bandera=0
    insta=[]
    instb=[]
    for i in intrs1:
        if bandera==1:
            break
        for j in intrs2:
            if (i.state==j.state):
                insta=[]
                instb=[]
                insta=[i]
                instb=[j]
                bandera=1
                break

    return insta,instb
def bidirectional_search(problem, frontierA, frontierP):

    frontierA.append(Node(problem.initial))
    frontierP.append(Node(problem.goal))

    nodos_visitadosA=[]
    nodos_visitadosP=[]

    exploredA = set()
    exploredP = set()

    nodos_en_memoriaA=1
    nodos_en_memoriaP=1

    lista_exploradosA=[]
    lista_exploradosP=[]
    
    listaVisitadosA=[]
    listaVisitadosP=[]

    while frontierA and frontierP:
        
        intrs1,intrs2=interseccion(lista_exploradosA,frontierP)#Si se cruzan en los explorados de A
        intrs3,intrs4=interseccion(frontierA,lista_exploradosP)#si se cruzan en los explorados de P
        intrs5,intrs6=interseccion(frontierA,frontierP)#si se cruzan solo en la frontera

        fronteraA=[]
        for i in frontierA:
            fronteraA.append(i.state)
        fronteraP=[]
        for i in frontierP:
            fronteraP.append(i.state)

        if len(intrs1)!=0:#primero vemos si sse cruzan en los explorados de A
            return intrs1[0], intrs2[0]
        if len(intrs3)!=0:#Luego vemos si sse cruzan en los explorados de P
            return intrs3[0], intrs4[0]
        if len(intrs5)!=0:# Al ultimo vemos si se estan cruzando las fronteras
            return intrs5[0], intrs6[0]

        nodeA = frontierA.pop()
        listaVisitadosA.append(nodeA.state)
        nodos_visitadosA.append(nodeA)
        aux1=[nodeA]
        lista_exploradosA+=aux1

        nodeP = frontierP.pop()
        nodos_visitadosP.append(nodeP)
        listaVisitadosP.append(nodeP.state)
        aux2=[nodeP]
        lista_exploradosP+=aux2

        exploredA.add(nodeA.state)
        exploredP.add(nodeP.state)

        frontierA.extend(   child for child in nodeA.expand(problem)
                            if( (child.state not in exploredA) and (child.state not in (ndo.state for ndo in frontierA)) )    )

        frontierP.extend(   child for child in nodeP.expand(problem)
                            if( (child.state not in exploredP) and (child.state not in (ndo.state for ndo in frontierP)) )    )

    return None,None

def combinarSoluciones(node_solucionBIS1,node_solucionBIS2):
    solucion1=node_solucionBIS1.solution()
    solucion2=node_solucionBIS2.solution()
    
    solv=[]
    cant=len(solucion2)
    for i in range(cant):
        c1=solucion2.pop()
        solv.append(c1)
        
    respuesta= solucion1+solv
    solucion3=[]
    for i in respuesta:
        if i not in solucion3:
            solucion3.append(i)

    return solucion3
#____________________________________________________________________________________________________________________________

import heapq
class FrontierPQ:
    """Una Frontera ordenada por una funcion de costo (Priority Queue)"""
    
    def __init__(self, initial, costfn=lambda node: node.path_cost):
        "Inicializa la Frontera con un nodo inicial y una funcion de costo especificada (por defecto es el costo de camino)."
        self.heap   = []
        self.states = {}
        self.costfn = costfn
        self.add(initial)
    
    def add(self, node):
        "Agrega un nodo a la frontera."
        cost = self.costfn(node)
        heapq.heappush(self.heap, (cost, node))
        self.states[node.state] = node
        
    def pop(self):
        "Remueve y retorna el nodo con minimo costo."
        (cost, node) = heapq.heappop(self.heap)
        self.states.pop(node.state, None) # remove state
        return node
    
    def replace(self, node):
        "node reemplaza al nodo de la Fontera que tiene el mismo estado que node."
        if node.state not in self:
            raise ValueError('{} no tiene nada que reemplazar'.format(node.state))
        for (i, (cost, old_node)) in enumerate(self.heap):
            if old_node.state == node.state:
                self.heap[i] = (self.costfn(node), node)
                heapq._siftdown(self.heap, 0, i)
                return

    def __contains__(self, state): return state in self.states
    
    def __len__(self): return len(self.heap)
"""
class cola_de_prioridad:

    def __init__(self, ,dato, costoFn=lambda node: node.path_cost):
        self.cola = queue.PriorityQueue()
"""
"""
def best_first_graph_search(problem, f):
    "" Busca el objetivo expandiendo el nodo de la frontera con el menor valor de la funcion f. Memoriza estados visitados
        Antes de llamar a este algoritmo hay que especificar La funcion f(node). Si f es node.depth tenemos Busqueda en Amplitud; 
        si f es node.path_cost tenemos Busqueda  de Costo Uniforme. Si f es una heurística tenemos Busqueda Voraz;
        Si f es node.path_cost + heuristica(node) tenemos A*
    ""

    #frontier = FrontierPQ( Node(problem.initial), f )  # frontera tipo cola de prioridad ordenada por f
    frontier = queue.PriorityQueue()
    frontier.put((f(Node(problem.initial)),Node(problem.initial)))

    explored = set()     # memoria de estados visitados
    visited_nodes = []   # almacena nodos visitados durante la busqueda
    while frontier:
        nodeT = frontier.get()  #retorn una tupla con el peso y su valor  e.g. (f, Node)
        node = nodeT[1] # Obtenemos el nodo de la tupla
        visited_nodes.append(node)        
        if problem.goal_test(node.state):
            return node, visited_nodes
        explored.add(node.state)
        for action in problem.actions(node.state):
            child = node.child_node(problem, action)
            peso = f(child)
            if child.state not in explored and child.state not in frontier:
                frontier.add(child)
            elif child.state in frontier:
                incumbent = frontier.states[child.state] 
                if f(child) < f(incumbent):
                    frontier.replace(child)
    return None,visited_nodes
"""

def memoize(fn, slot=None, maxsize=32):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, use lru_cache for caching the values."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn

class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first.
    If order is 'min', the item with minimum f(x) is
    returned first; if order is 'max', then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order='min', f=lambda x: x):
        self.heap = []

        if order == 'min':
            self.f = f
        elif order == 'max':  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("order must be either 'min' or 'max'.")

    def append(self, item):
        """Insert item at its correct position."""
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item (with min or max f(x) value)
        depending on the order."""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.heap)

    def __contains__(self, key):
        """Return True if the key is in PriorityQueue."""
        return any([item == key for _, item in self.heap])

    def __getitem__(self, key):
        """Returns the first value associated with key in PriorityQueue.
        Raises KeyError if key is not present."""
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        """Delete the first occurrence of key."""
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)


def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None

def astar_search(problem, heuristic):
    """Algoritmo A*, un caso especial de best_first_graph_search con f = path_cost + heuristic"""
    f = lambda node: node.path_cost + heuristic(node, problem)
    return best_first_graph_search(problem, f)

def nullheuristic(node, problem):   
    return 0

def h1(node, problem):
    hrstca = problem.heuristica
    #print(hrstca[int(node.state)])
    return hrstca[int(node.state)]
#____________________________________________________________________________________________________________________________