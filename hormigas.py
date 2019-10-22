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
        if ch != 'S' and ch != 'E': a[col][row] = (0,255,255)
    
    for node in solutionNodes:  
        row,col = node.state
        ch = maze.grid[row][col]
        if ch != 'S' and ch != 'E': a[col][row] = (0,0,255)
          
    
    
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
"""
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

w,h=im.size
print(w,h)
instancia=[]
for y in range(h):
  temp = []
  for x in range(w):
    cpixel=pix[x,y]
    if cpixel==(0,255,0): costo=1
    elif cpixel==(0,0,255): costo=3
    elif cpixel==(255,255,255): costo=2
    elif cpixel==(255,0,0): costo=1
    temp.append(costo)
  instancia.append(temp)

feromonas=[[0 for i in range(0,w)]for i in range(0,h)]




class aco_TSP:
    def __init__(self):
    
        self.instancia = instancia
        self.feromonas = feromonas
        self.problem=p
        self.alfa = 1.0
        self.beta = 1.0
        self.P = 0.5
        self.Tij = 1.0
        self.ciudades = 0
        self.hormigas = 1
        self.pos_hormiga = self.problem.initial
        self.pos_anterior = 1
        self.recorridos = [] 
        self.sitios = [0]
   
    def correrHormiga(self):
        gral = []
        for hormigas in range(self.hormigas):
          #gral.append(self.problem.initial)
          #i=0
            r =  self.seleccionarArista() #mando a correr a una hormiga y almacen en r su recorrido
            #gral.append(r)#mando los recorridos a una arreglo general
            #self.actualizarFeromonas(r) #al final de cada hormiga actualizo la feromona
        print ("Recorridos de cada hormiga: ",gral)
 
        #self.evaluarRecorridos(gral) #Evaluo los recorridos de cada Hormiga

  
    def seleccionarArista(self):
      
      
      pos_hormiga = self.pos_hormiga
      
      nodo=Node(self.pos_hormiga)
      
      sumatoria = 0
      Tij_and_Nij = []
      pos_ant = self.pos_hormiga
      pos_act = self.pos_hormiga
      aristas = []
      recorridos=[]
      
      while((self.problem.goal_test(nodo.state)) ==False):
        aristas = []
        for action in (self.problem.actions(nodo.state)):

          child = nodo.child_node(self.problem, action)
          
          #if not child.state in recorridos:
          aristas.append(child.state)
            
        print(pos_act)

        Tij_and_Nij = []
        menor = -1
        for elemento in aristas: #sacar la sumatoria
                        #calcular tij and nij

          a,l=elemento
          nodo=Node(elemento)
          Tij = self.feromonas[a][l]
          Nij = 1.0/(self.instancia[a][l]+manhatan_dist(nodo, self.problem))
          elev = math.pow((1+Tij),self.alfa)*math.pow(Nij,self.beta)
          Tij_and_Nij.append(elev)
          if menor==-1:
            menor = elev
          else:
            menor = min(menor,elev)		
        i=0
        for elemento in aristas:
          if elemento in recorridos:
              Tij_and_Nij[i]=menor/1000
          i+=1
        #print(Tij_and_Nij)
        sumatoria = 0
        for elemento in Tij_and_Nij:    
          sumatoria += elemento
          
        prob = np.random.rand(1)*sumatoria
        i=0
        for T_N in Tij_and_Nij:#asignar a ruleta en base a probabilidad
          sumatoria -= T_N
          if sumatoria<=prob:
            arista = aristas[i]
            break
          i+=1
        #print(nodo.state)
        #print(ruleta)

              #print(arista)
        pos_ant = pos_act
        pos_act = arista
        recorridos.append(pos_act)
        #print(recorridos)
        #self.ponerFeromonas(pos_ant,pos_act)
        #print(recorridos)
        nodo=Node(pos_act)
      
      
      print(recorridos)
      recorridos.append(0)
      return recorridos     
    
    def ponerFeromonas(self,pos_anterior,pos_hormiga):
      
      (wa,ha)=pos_anterior
      wn,hn=pos_hormiga
      #print(pos_anterior)
      self.feromonas[wa][ha] += self.Tij
      self.feromonas[wn][hn] += self.Tij

            
    def sumaTij_K(self,recorridos):
        suma = 0
        #for i in range(self.ciudades):
        L_k = 0
        for elemento in recorridos:
            w,h=elemento
            costo=self.instancia[w][h]
            L_k += costo
        suma += 1/L_k
        return suma
            
    def actualizarFeromonas(self,recorridos):
         
        sumaT_i_j = self.sumaTij_K(recorridos)
        nuevas_feromonas = []
        for i in range(len(self.feromonas)):
            lote = []
            for elemento in self.feromonas[i]:
                lote.append( (1-self.P)*elemento + sumaT_i_j )
            nuevas_feromonas.append(lote)
 
        self.feromonas = nuevas_feromonas
  
    def evaluarRecorridos(self,recorridos):
        index = 0
        evaluacion_ant = 0
        for i in range(len(recorridos)):
            evaluacion = 0
            elem_ant = recorridos[i][0]
            for elemento in recorridos[i]:
                #if elemento != 0:
                w,h=elemento
                evaluacion += self.instancia[w][h]
                elem_ant = elemento
             
            if evaluacion < evaluacion_ant:
                index = i
                evaluacion_ant = evaluacion
            if evaluacion_ant == 0:
                evaluacion_ant = evaluacion
 
            print ("recorrido Ant =",i," evaluacion: ",evaluacion)
 
        print ("\nEl optimo es con: ",recorridos[index])
     

  
aco = aco_TSP()
aco.correrHormiga()













