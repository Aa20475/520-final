from __future__ import annotations

import argparse
from copy import deepcopy
import numpy as np
from enum import Enum
import random
from collections import deque
from queue import PriorityQueue
from IPython.display import clear_output
from tqdm import tqdm

# custom print function
def printStuff(*args, log = False , end = "\n"):
    global debug
    if log:
        if debug:
            print(args,end=end)
    else:
        if not debug:
            print(args,end=end)

# Enum Class for possible moves
class Move(Enum):
    UP = (-1,0)
    LEFT = (0,-1)
    DOWN = (1,0)
    RIGHT = (0,1)

    @staticmethod
    def getOpp(move: Move):
        if move==Move.RIGHT:
            return Move.LEFT
        elif move==Move.UP:
            return Move.DOWN
        elif move==Move.DOWN:
            return Move.UP
        elif move==Move.LEFT:
            return Move.RIGHT
        
# Argument parsing logic
def buildArgs():
    parser = argparse.ArgumentParser(description="Script to save the Nuclear power station")
    
    parser.add_argument("--schema",required=False,default="./Thor23-SA74-VERW-Schematic (Classified).txt",help="Path of the schema file")
    parser.add_argument("--generate",required=False,default=False, type=bool, help="if True, generates a random map ignoring schema argument")
    parser.add_argument("--rows",required=False,default=5, type=int, help="Rows used during map generation")
    parser.add_argument("--columns",required=False,default=5, type=int, help="Columns used during map generation")
    parser.add_argument("--algo",required=False,default=0, type=int, help="If 1, runs greedy approach")
    parser.add_argument("--debug",required=False,default=0, type=bool, help="If True, prints debug stuff")

    return parser.parse_args()

# Converts a single line of string into array of schema
def strToSchema(s:str):
    return [0 if c=='X' else 1 for c in s.split('\n')[0] ]

# checks if a position is valid
def isValid(schema : np.ndarray,i,j):
    return (i>=0 and i<schema.shape[0] and j>=0 and j<schema.shape[1] and schema[i][j])

# Performs belief update based on a give move 
# Note that it creates a new copy of beliefs
def beliefUpdateStep(schema: np.ndarray, beliefs :  np.ndarray, move : Move):
    newBeliefs = np.zeros(beliefs.shape)
    for i in range(0,beliefs.shape[0]):
        for j in range(0,beliefs.shape[1]):
            if schema[i][j]:
                # update belief for unblocked cells
                newI = i+ move.value[0]
                newJ = j+ move.value[1]
                if isValid(schema, newI,newJ):
                    newBeliefs[newI][newJ] += beliefs[i][j]
                else:
                    newBeliefs[i][j] += beliefs[i][j]
    return newBeliefs

def getHeuristic(p: np.ndarray):
    minY, maxY= p.shape[1],0
    minX, maxX= p.shape[0],0
    for i in range(0,p.shape[0]):
        for j in range(0,p.shape[1]):
            if p[i][j]!=0:
                minX, maxX =  min(minX,i),max(maxX,i)
                minY, maxY =  min(minY,j),max(maxY,j)
    
    return abs(minX-maxX)+abs(maxX-maxY)
    

def getHeuristicOld(p:np.ndarray):
    maxVal = np.max(p)
    listKeys = []
    for i in range(0,p.shape[0]):
        for j in range(0,p.shape[1]):
            if p[i][j]==maxVal:
                listKeys.append((i,j))
    if len(listKeys)==1:
        secondMax = np.max(p[p!=maxVal])
        listKeysSecond = []
        for i in range(0,p.shape[0]):
            for j in range(0,p.shape[1]):
                if p[i][j]==secondMax:
                    listKeysSecond.append((i,j))
        h = 0
        for i in range(0,len(listKeysSecond)):
            h = max(h, max(listKeys[0][0],listKeysSecond[i][0])+max(listKeys[0][1],listKeysSecond[i][1]))        
        return h
    h = 0
    for i in range(0,len(listKeys)):
        for j in range(0,len(listKeys)):
            if i!=j:
                h = max(h, max(listKeys[i][0],listKeys[j][0])+max(listKeys[i][1],listKeys[j][1]))
    return h

def simulateGame(schema,moves):
    beliefs = schema/np.count_nonzero(schema==1)
    # print(beliefs)

    for move in moves:
        # print("Move :",move.name," ===========================")
        beliefs = beliefUpdateStep(schema,beliefs,move)
    #     print(beliefs)
    return beliefs
    # print("===========================")

# Performs greedy search with random tie-breaking
def getGreedyMoveSequence(schema):
    beliefs = schema/np.count_nonzero(schema==1)

    totalTiles= beliefs.shape[0] * beliefs.shape[1]
    # printStuff("Total tiles: ",totalTiles.)
    bestMovesSequence = []

    while np.count_nonzero(beliefs==0.0)!= totalTiles-1:
        tmpBeliefs = {}
        for move in Move:
            tmpBeliefs[move] = beliefUpdateStep(schema,beliefs,move)
        
        zeros = [np.count_nonzero(val==0.0) for val in tmpBeliefs.values()]
        mx = max(zeros)

        keyList = list(tmpBeliefs.keys())
        bestMoves = [keyList[i] for i in range(0,len(zeros)) if zeros[i]==mx ]

        bestMove = random.choice(bestMoves)
        beliefs = tmpBeliefs[bestMove]
        bestMovesSequence.append(bestMove)
        # printStuff("Made move ",bestMove.name)

    return bestMovesSequence

def getBFSMovesSequence(schema):
    global debug
    # initial beliefs
    beliefs = schema/np.count_nonzero(schema==1)

    fringe = deque()
    fringe.append(([],beliefs))

    totalTiles= beliefs.shape[0] * beliefs.shape[1]
    printStuff("Total tiles: ",totalTiles)
    printStuff("============")

    
    bestPath = None

    while len(fringe)!=0:
        printStuff("[Fringe size : %d] [bestPath len : %d]"%(len(fringe), 0 if bestPath is None else len(bestPath)),end="\r")
        seq,beliefs = fringe.popleft()

        # goal state
        if  np.count_nonzero(beliefs==0.0)== totalTiles-1:
            # goal state
            if bestPath is None:
                bestPath = seq
            if len(bestPath)<len(seq):
                bestPath = seq
            # printStuff("Found Goal!")
            break
            

        tmpBeliefs = {}
        zeros = {}
        for move in Move:
            tmpBeliefs[move] = beliefUpdateStep(schema,beliefs,move)
            zeros[move] = np.count_nonzero(tmpBeliefs[move]==0.0)
        
        mx = max(zeros.values())

        bestMoves = [move for move in zeros if zeros[move]==mx]
        
        if debug:
            printStuff("BestMove: ",bestMoves,log=True)
            printStuff("====",log=True)
        if len(bestMoves)==1:
            fringe.append((seq+[bestMoves[0]],tmpBeliefs[bestMoves[0]]))
        else:
            for move in bestMoves:
                if len(seq)==0 or (len(seq)!=0 and move!=seq[-1]):
                    fringe.append((seq+[move],tmpBeliefs[move]))
        if debug:
            printStuff("Fringe",log=True)
            for f in fringe:
                printStuff(f[0],log=True)
            printStuff("============",log=True)

    return bestPath if not bestPath is None else []

def getDijkstraMovesSequence(schema : np.ndarray):
    global debug

    totalTile = schema.shape[0]*schema.shape[1]

    beliefs = schema / np.count_nonzero(schema==1)

    fringe = PriorityQueue()
    fringe.put((0,beliefs.tobytes()))

    costs : dict = {}
    costs[beliefs.tobytes()] = 0
    
    pathStore = {}
    pathStore[beliefs.tobytes()] = None

    greedyPath = getGreedyMoveSequence(schema)
    endState = simulateGame(schema,greedyPath).tobytes()
    costs[endState] = len(greedyPath)
    def getPath(stat):
        path = []
        current = pathStore[stat]
        while not current is None:
            path.append(current[1])
            current = pathStore[current[0]]
        path.reverse() # optional
        return path


    pruned = 0
    while not fringe.empty():
        pr,curr = fringe.get()
        currArray = np.frombuffer(curr,dtype=beliefs.dtype).reshape(beliefs.shape)
        printStuff("[Fringe size : %d] [bestPath len : %d] [pruned: %d] [currLen: %d] [estimate: %d] "%(fringe.qsize(), costs[endState],pruned,costs[curr],pr),end="\r")

        # print("------- curr [%0.4f] --------------------"%pr)
        # print(currArray)
        if costs[curr]>costs[endState]:
            pruned+=1
            continue
        if pr>costs[endState]:
            pruned+=1
            continue
        

        if np.count_nonzero(currArray==0.0)==totalTile - 1:
            # print("Goal!")
            if endState is None:
                endState = curr
            if costs[endState]>costs[curr]:
                endState = curr
            continue
        
        tmpBeliefs = {}
        zeros = {}
        for move in Move:
            tmpBeliefs[move] = beliefUpdateStep(schema,currArray,move)
            zeros[move] = np.count_nonzero(tmpBeliefs[move]==0.0)

        for move in Move:
            probs = tmpBeliefs[move]
            probString = probs.tobytes()
            newCost = costs[curr] + 1
            if probString not in costs or newCost < costs[probString]:
                # print(probs)
                # print("Added ",move.name,"====>",newCost)
                costs[probString] = newCost
                fringe.put((newCost+getHeuristic(probs),probs.tobytes()))

                pathStore[probString] = (curr,move)
    print()
    print(np.frombuffer(endState,dtype=beliefs.dtype).reshape(beliefs.shape))
    if endState not in pathStore:
        print("Greedy")
        return greedyPath
    return [] if endState is None else getPath(endState)

if __name__ =="__main__":

    args = buildArgs()

    debug = args.debug

    schema = []
    if not args.generate:
        with open(args.schema,"r") as f:
            schema = [strToSchema(x) for x in f.readlines()]
    
    schema = np.array(schema)
    if args.algo==1:
        bestMovesSequence = getGreedyMoveSequence(schema)
    elif args.algo==2:
        bestMovesSequence = getBFSMovesSequence(schema)
    elif args.algo==0:
        bestMovesSequence = getDijkstraMovesSequence(schema)

    clear_output(wait=True)
    # print(bestMovesSequence)
    
    # simulateGame(schema,bestMovesSequence)
    printStuff("Length of Path: ",len(bestMovesSequence))
    printStuff("Best moves: ",[i.name for i in bestMovesSequence])