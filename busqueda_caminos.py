from PIL import Image
from numpy import array
from time import time
import numpy as np
import random
import cv2
import sys
import argparse

class Maze:

    def __init__(self, grid):
        """ Construye el maze a partir del grid ingresado
            grid: debe ser una matriz (lista de listas), ejemplo [['#','S',' '],['#',' ','E']]  """
        self.grid = grid
        self.numRows = len(grid)
        self.numCols = len(grid[0])
        flag=0
        for i in range(self.numRows):
            for j in range(self.numCols):
                if len(grid[i]) != self.numCols:
                    raise "Grid no es Rectangular"
                if grid[i][j] == (255,0,0) and flag==0:
                    self.startCell = (i,j)
                    flag = 1
                if grid[i][j] == (255,0,0):
                    self.exitCell= (i,j)
        if self.exitCell == None:
            raise "No hay celda de Inicio"
        if self.startCell == None:
            raise "No hay celda de salida"
   
    def isPassable(self, row, col):
 
        return self.isWhite(row, col) or self.isRed(row,col) or self.isGreen(row, col) or self.isBlue(row, col)
      
    def isWhite(self, row, col):
        
        return self.grid[row][col] == (255,255,255)
    
    def isRed(self, row, col):
       
        return self.grid[row][col] == (255,0,0)
      
    def isGreen(self, row, col):
     
        return self.grid[row][col] == (0,255,0)
      
    def isBlue(self, row, col):
     
        return self.grid[row][col] == (0,0,255)      
    
    def isBlocked(self, row,col):
   
        return self.grid[row][col] == (0,0,0)   
        
    def getNumRows(self):
        """ Retorna el numero de filas en el maze """
        return self.numRows
  
    def getNumCols(self):
        """ Retorna el numero de columnas en el maze """
        return self.numCols  
   
    def getStartCell(self):
        """ Retorna la posicion (row,col) de la celda de inicio """
        return self.startCell
  
    def getExitCell(self):
        """ Retorna la posicion (row,col) de la celda de salida """
        return self.exitCell

    def __getAsciiString(self):
        """ Retorna el string de vizualizacion del maze """
        lines = []
        headerLine = ' ' + ('-' * (self.numCols)) + ' '
        lines.append(headerLine)
        for row in self.grid:
            rowLine = '|' + ''.join(row) + '|'
            lines.append(rowLine)
        lines.append(headerLine)
        return '\n'.join(lines)

    def __str__(self):
        return self.__getAsciiString()


def readMazeFromFile(file):
    """ Lee un archivo que contiene un laberinto y retorna una instancia de Maze con dicho laberinto"""
    im = Image.open(file) # Can be many different formats.
    pix = im.load()

    type(pix)

    w,h=im.size
    all_pixel=[]
    for y in range(h):
      temp = []
      for x in range(w):
        cpixel=pix[x,y]
        temp.append(cpixel)
      all_pixel.append(temp)
       
    return Maze(all_pixel)

class SearchProblem(object):
    def __init__(self, initial, goal=None):
        """Este constructor especifica el estado inicial y posiblemente el estado(s) objetivo(s),
        La subclase puede anadir mas argumentos."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Retorna las acciones que pueden ser ejecutadas en el estado dado.
        El resultado es tipicamente una lista."""
        raise NotImplementedError

    def result(self, state, action):
        """Retorna el estado que resulta de ejecutar la accion dada en el estado state.
        La accion debe ser alguna de self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Retorna True si el estado pasado satisface el objetivo."""
        raise NotImplementedError

    def path_cost(self, c, state1, action, state2):
        """Retorna el costo del camino de state2 viniendo de state1 con 
        la accion action, asumiendo un costo c para llegar hasta state1. 
        El metodo por defecto cuesta 1 para cada paso en el camino."""
        return c + 1


class MazeSearchProblem(SearchProblem):
    def __init__(self, maze):
        """El constructor recibe el maze"""
        self.maze = maze
        self.initial = maze.getStartCell()
        self.goal = maze.getExitCell()
        self.numNodesExpanded = 0        
        self.expandedNodeSet = {}   
        
    def __isValidState(self,state):
        """ Retorna true si el estado dado corresponde a una celda no bloqueada valida """
        row,col = state
        if row < 0 or row >= self.maze.getNumRows():
            return False
        if col < 0 or col >= self.maze.getNumCols():
            return False
        return not self.maze.isBlocked(row,col)        

    def actions(self, state):
        """Retorna las acciones legales desde la celda actual """
        row,col = state
        acciones = []
        if self.__isValidState((row-1, col)):
            acciones.append('N')

        ## TO DO: Completar

        if self.__isValidState((row+1, col)):
            acciones.append('S')
            
        if self.__isValidState((row, col+1)):
            acciones.append('O')
            
        if self.__isValidState((row, col-1)):
            acciones.append('E')
        
        
        ##################       

        return acciones
    
    def result(self, state, action):
        """Retorna el estado que resulta de ejecutar la accion dada desde la celda actual.
        La accion debe ser alguna de self.actions(state)"""  
        row,col = state
        newState = ()
        if action == 'N':
            newState = (row-1, col)
        ## TO DO: Completar
        
        
        if action == 'S':
            newState = (row+1, col)
            
        
        if action == 'O':
            newState = (row, col+1)
            
        
        if action == 'E':
            newState = (row, col-1)    
        
        ##################      
        return newState
        
    def goal_test(self, state):
        """Retorna True si state es self.goal"""
        return (self.goal == state) 

    def path_cost(self, c, state1, action, state2):
        """Retorna el costo del camino de state2 viniendo de state1 con la accion action 
        El costo del camino para llegar a state1 es c. El costo de la accion sale de self.maze """
        row, col = state2
        actionCost = None

        ## TO DO: Completar
        ## Defina el valor de la variable actionCost, de acuerdo al tipo de celda (con agua, 
        ## con arena o vacia). Por ejemplo, si es una celda con agua, el costo de la accion
        ## es 5. Para esta parte de la implementacion, no se requiere verificar que la celda
        ## sea una celda con obstaculo ('#').
        
        (x,y)=state2
        
        Aux=self.maze.grid[x][y]
        
        actionCost=0
                
        if Aux==(0,255,0): actionCost=1
        #elif Aux==(0,0,0): actionCost=1000000
        elif Aux==(0,0,255): actionCost=3
        elif Aux==(255,255,255): actionCost=2
        elif Aux==(255,0,0): actionCost=1
        
        ##################
        
        if state2 == self.maze.getStartCell() or state2 == self.maze.getExitCell():
            actionCost = 1
        
        return c + actionCost



class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Crea un nodo de arbol de busqueda, derivado del nodo parent y accion action"
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def expand(self, problem):
        "Devuelve los nodos alcanzables en un paso a partir de este nodo."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state, action, next))

    def solution(self):
        "Retorna la secuencia de acciones para ir de la raiz a este nodo."
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Retorna una lista de nodos formando un camino de la raiz a este nodo."
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))
    
    def __lt__(self, node):
        return self.state < node.state
    
    def __eq__(self, other): 
        "Este metodo se ejecuta cuando se compara nodos. Devuelve True cuando los estados son iguales"
        return isinstance(other, Node) and self.state == other.state
    
    def __repr__(self):
        return "<Node {}>".format(self.state)
    
    def __hash__(self):
        return hash(self.state)


from collections import deque

class FIFOQueue(deque):
    """Una cola First-In-First-Out"""
    def pop(self):
        return self.popleft()


import heapq
class FrontierPQ:
    "Una Frontera ordenada por una funcion de costo (Priority Queue)"
    
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



def graph_search(problem, frontier):
    frontier.append(Node(problem.initial))
    explored = set()     # memoria de estados visitados
    visited_nodes = []   # almacena nodos visitados durante la busqueda
    while frontier:
        node = frontier.pop()
        visited_nodes.append(node)
        if problem.goal_test(node.state):
            return node, visited_nodes
        explored.add(node.state)
        
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and
                        child not in frontier)
    return None



def best_first_graph_search(problem, f):
    """Busca el objetivo expandiendo el nodo de la frontera con el menor valor de la funcion f. Memoriza estados visitados
    Antes de llamar a este algoritmo hay que especificar La funcion f(node). Si f es node.depth tenemos Busqueda en Amplitud; 
    si f es node.path_cost tenemos Busqueda  de Costo Uniforme. Si f es una heuristica tenemos Busqueda Voraz;
    Si f es node.path_cost + heuristica(node) tenemos A* """

    frontier = FrontierPQ( Node(problem.initial), f )  # frontera tipo cola de prioridad ordenada por f
    explored = set()     # memoria de estados visitados
    visited_nodes = []   # almacena nodos visitados durante la busqueda
    while frontier:
        node = frontier.pop()
        visited_nodes.append(node)        
        if problem.goal_test(node.state):
            return node, visited_nodes
        explored.add(node.state)
        for action in problem.actions(node.state):
            child = node.child_node(problem, action)
            if child.state not in explored and child.state not in frontier:
                frontier.add(child)
            elif child.state in frontier:
                incumbent = frontier.states[child.state] 
                if f(child) < f(incumbent):
                    frontier.replace(child)

def astar_search(problem, heuristic):
    f = lambda node: node.path_cost + heuristic(node, problem)
    return best_first_graph_search(problem, f)

def nullheuristic(node, problem):   
    return 0



#Para elevar un numero a una potencia, se sugiere usar la libreria math
import math
math.pow(4,2)


def straightline_dist(node, problem):
    "Distancia en linea recta desde la celda de node hasta la celda Objetivo (problem.goal)"
    ## TODO: Completar
    (nx,ny)=node.state
    (gx,gy)=problem.maze.getExitCell()
    
    return ((nx-gx)**2+(ny-gy)**2)**0.5
  
def manhatan_dist(node, problem):
    "Distancia Manhatan (o city block) desde la celda de node hasta la celda Objetivo (problem.goal)"
    ## TODO: Completar
    
    (nx,ny)=node.state
    (gx,gy)=problem.maze.getExitCell()
    
    return abs(nx-gx)+abs(ny-gy)


def displayResults(file,maze, visitedNodes, solutionNodes,nombre):
    """ Muestra los resultados de busqueda en el maze.   """
    
    grid_copy = []
    for row in maze.grid:
        grid_copy.append([x for x in row]) 
        
    a=cv2.imread(file,cv2.IMREAD_COLOR)
    
    for node in visitedNodes:  
        row,col = node.state
        ch = maze.grid[row][col]
        if ch != 'S' and ch != 'E': a[row][col] = (0,255,255)
    
    for node in solutionNodes:  
        row,col = node.state
        ch = maze.grid[row][col]
        if ch != 'S' and ch != 'E': a[row][col] = (0,0,255)
          
    
    
    cv2.imwrite(nombre,a)





#start_time=time()
#nsol, visited_nodes = astar_search(p, nullheuristic)
#duracion=time()-start_time
#print('Solucion A* y heuristica nula (UCS):Nodos solucion={} Nodos visitados={}. Costo Solucion = {}  Duracion = {}'.format(len(nsol.solution()), len(visited_nodes),nsol.path_cost,duracion))
#displayResults(mazeFile,maze, visited_nodes, nsol.path(),'UCS.png')

#start_time=time()
#nsol, visited_nodes = astar_search(p, straightline_dist)
#duracion=time()-start_time
#print('Solucion A* y heuristica straightline_dist:Nodos solucion={} Nodos visitados={}. Costo Solucion = {}  Duracion = {}'.format(len(nsol.solution()), len(visited_nodes),nsol.path_cost,duracion))
#displayResults(mazeFile,maze, visited_nodes, nsol.path(),'ASTAR1.png')

#start_time=time()
#nsol, visited_nodes = astar_search(p, manhatan_dist)
#duracion=time()-start_time
#print('Solucion A* y heuristica manhattan:Nodos solucion={} Nodos visitados={}. Costo Solucion = {}  Duracion = {}'.format(len(nsol.solution()), len(visited_nodes),nsol.path_cost,duracion))
#displayResults(mazeFile,maze, visited_nodes, nsol.path(),'ASTAR2.png')




parser = argparse.ArgumentParser()
parser.add_argument("-i")
parser.add_argument("-m")
parser.add_argument("-a")
args = parser.parse_args()


if args.i != None:
    mazeFile = args.i
    
""" Carga un laberinto de archivo en disco e instancia el problema de busqueda.   """

im = Image.open(mazeFile) # Can be many different formats.
pix = im.load()

type(pix)

w,h=im.size
#print(w,h)
all_pixel=[]
for x in range(w):
  for y in range(h):
    cpixel=pix[x,y]
    all_pixel.append(cpixel)

type(all_pixel)

#mazeFile='a.bmp'

maze = readMazeFromFile(mazeFile)
#visited_nodes=[]
#displayResults(maze, visited_nodes, None,'b.bmp')
p = MazeSearchProblem(maze)
#print(maze)
#file='a.bmp'

maze = readMazeFromFile(mazeFile)
#genetic_search_findtour(maze,'a.bmp',1,1,0.9,"ox","swap")

if args.m != None and args.m == 'ucs': #solo se ha implementado el algoritmo genetico
  start_time=time()
  nsol, visited_nodes = astar_search(p, nullheuristic)
  duracion=time()-start_time
  print('Solucion A* y heuristica nula (UCS):Nodos solucion={} Nodos visitados={}. Costo Solucion = {}  Duracion = {}'.format(len(nsol.solution()), len(visited_nodes),nsol.path_cost,duracion))
  displayResults(mazeFile,maze, visited_nodes, nsol.path(),'UCS.png')
  sys.exit()
    
if args.m != None and args.m == 'astar': #solo se ha implementado el algoritmo genetico
  start_time=time()
  nsol, visited_nodes = astar_search(p, manhatan_dist)
  duracion=time()-start_time
  print('Solucion A* y heuristica manhattan:Nodos solucion={} Nodos visitados={}. Costo Solucion = {}  Duracion = {}'.format(len(nsol.solution()), len(visited_nodes),nsol.path_cost,duracion))
  displayResults(mazeFile,maze, visited_nodes, nsol.path(),'ASTAR2.png')
  sys.exit()

"""  
if args.m != None and args.m == 'aco' and args.a!=None:
  param=args.a.split(',')
    for p in param:
      if p.startswith('alpha='):
            alpha = float(p[6:])
      elif p.startswith('betha='):
            mut_rate = float(p[6:]) 
      elif p.startswith('rho='):
            cx_op = float(p[4:])
      elif p.startswith('numAnts='):
            w = int(p[8:]) 
  
    aco(maze, mazeFile, w, max_gener, mut_rate, cx_op, mut_op)
"""
