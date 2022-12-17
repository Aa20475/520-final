from __future__ import annotations

import argparse
from copy import deepcopy
import numpy as np
from enum import Enum
import random
from collections import deque
from IPython.display import clear_output

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

# Performs greedy search with random tie-breaking
def getGreedyMoveSequence(schema):
    beliefs = schema/np.count_nonzero(schema==1)

    totalTiles= beliefs.shape[0] * beliefs.shape[1]
    printStuff("Total tiles: ",totalTiles)
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
        bestMovesSequence.append(bestMove.name)
        # printStuff("Made move ",bestMove.name)

    return bestMovesSequence

def getBestMovesSequence(schema):
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
    elif args.algo==0:
        bestMovesSequence = getBestMovesSequence(schema)

    clear_output(wait=True)
    printStuff([i.name for i in bestMovesSequence])
    printStuff("Number of move: ",len(bestMovesSequence))