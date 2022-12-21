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
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'

import pygame


import math
from time import sleep, time

# custom print function
def printStuff(*args, log=False, end="\n"):
    global debug
    if log:
        if debug:
            print(args, end=end)
    else:
        if not debug:
            print(args, end=end)


# Enum Class for possible moves
class Move(Enum):
    UP = (-1, 0)
    LEFT = (0, -1)
    DOWN = (1, 0)
    RIGHT = (0, 1)

    @staticmethod
    def getOpp(move: Move):
        if move == Move.RIGHT:
            return Move.LEFT
        elif move == Move.UP:
            return Move.DOWN
        elif move == Move.DOWN:
            return Move.UP
        elif move == Move.LEFT:
            return Move.RIGHT


# Argument parsing logic
def buildArgs():
    parser = argparse.ArgumentParser(
        description="Script to save the Nuclear power station"
    )

    parser.add_argument(
        "--schema",
        required=False,
        default="./Thor23-SA74-VERW-Schematic (Classified).txt",
        help="Path of the schema file",
    )
    parser.add_argument(
        "--generate",
        required=False,
        default=False,
        type=bool,
        help="if True, generates a random map ignoring schema argument",
    )
    parser.add_argument(
        "--rows",
        required=False,
        default=5,
        type=int,
        help="Rows used during map generation",
    )
    parser.add_argument(
        "--columns",
        required=False,
        default=5,
        type=int,
        help="Columns used during map generation",
    )
    parser.add_argument(
        "--algo", required=False, default=0, type=int, help="If 1, runs greedy approach"
    )
    parser.add_argument(
        "--debug",
        required=False,
        default=0,
        type=bool,
        help="If True, prints debug stuff",
    )
    parser.add_argument(
        "--ui",
        required=False,
        default=0,
        type=bool,
        help="If True, runs UI for greedy approach",
    )

    return parser.parse_args()

def runUI(schema, moves):
    pygame.init()
    screen = pygame.display.set_mode([900, 900])
    colors = {
        "WHITE": (255, 255, 255),
        "BLACK": (0, 0, 0),
        "RED": (255, 0, 0),
    }
    beliefs = np.array(schema != 0, dtype=int) / np.count_nonzero(schema == 1)

    boardLimits = [800, 800]

    agentPos = (0, 0)
    for i in range(0, schema.shape[0]):
        for j in range(0, schema.shape[1]):
            if schema[i][j] == 2:
                agentPos = (i, j)
                break

    box = 10
    font = pygame.font.Font("freesansbold.ttf", math.floor(box))

    startCenter = [50, 50]
    if schema.shape[0] > schema.shape[1]:
        box = boardLimits[0] // schema.shape[0]
        startCenter[0] += box * (schema.shape[0] - schema.shape[1]) // 2
    else:
        box = boardLimits[1] // schema.shape[1]
        startCenter[1] += box * (schema.shape[1] - schema.shape[0]) // 2

    def updateSchema(schema, agentPos, move):

        # print(move)
        i, j = agentPos[0] + move.value[0], agentPos[1] + move.value[1]
        if isValid(schema, i, j):
            schema[agentPos[0], agentPos[1]] = 1
            schema[i, j] = 2
            # print(agentPos,"-> ",(i,j))
            return (i, j)
        return agentPos

    for k in range(0, len(moves)):
        screen.fill(colors["WHITE"])
        sleep(0.5)
        pygame.draw.rect(
            screen,
            colors["BLACK"],
            (
                startCenter[0],
                startCenter[1],
                schema.shape[1] * box,
                schema.shape[0] * box,
            ),
            width=2,
        )
        for i in range(0, schema.shape[0]):
            for j in range(0, schema.shape[1]):
                if schema[i][j] == 0:
                    pygame.draw.rect(
                        screen,
                        colors["BLACK"],
                        (startCenter[0] + j * box, startCenter[1] + i * box, box, box),
                    )
                elif schema[i][j] == 2:
                    pygame.draw.rect(
                        screen,
                        colors["RED"],
                        (startCenter[0] + j * box, startCenter[1] + i * box, box, box),
                    )
                elif schema[i][j] == 1:
                    pygame.draw.rect(
                        screen,
                        (255, 255 * beliefs[i][j], 255),
                        (startCenter[0] + j * box, startCenter[1] + i * box, box, box),
                    )

        for i in range(0, schema.shape[0]):
            for j in range(0, schema.shape[1]):
                if schema[i][j]:
                    text = font.render(
                        ("%0.3f" % (beliefs[i][j])), False, colors["BLACK"]
                    )
                    textRect = text.get_rect()
                    textRect.center = (
                        startCenter[0] + j * box + box / 2,
                        startCenter[1] + i * box + box / 2,
                    )
                    screen.blit(text, textRect)

        pygame.display.flip()
        beliefs = beliefUpdateStep(schema, beliefs, moves[k])
        agentPos = updateSchema(schema, agentPos, moves[k])
        pygame.draw.rect(
            screen,
            colors["BLACK"],
            (
                startCenter[0],
                startCenter[1],
                schema.shape[1] * box,
                schema.shape[0] * box,
            ),
            width=2,
        )
        for i in range(0, schema.shape[0]):
            for j in range(0, schema.shape[1]):
                if schema[i][j] == 0:
                    pygame.draw.rect(
                        screen,
                        colors["BLACK"],
                        (startCenter[0] + j * box, startCenter[1] + i * box, box, box),
                    )
                elif schema[i][j] == 2:
                    pygame.draw.rect(
                        screen,
                        colors["RED"],
                        (startCenter[0] + j * box, startCenter[1] + i * box, box, box),
                    )
                elif schema[i][j] == 1:
                    pygame.draw.rect(
                        screen,
                        (255, 255 * beliefs[i][j], 255),
                        (startCenter[0] + j * box, startCenter[1] + i * box, box, box),
                    )

        for i in range(0, schema.shape[0]):
            for j in range(0, schema.shape[1]):
                if schema[i][j]:
                    text = font.render(
                        ("%0.3f" % (beliefs[i][j])), False, colors["BLACK"]
                    )
                    textRect = text.get_rect()
                    textRect.center = (
                        startCenter[0] + j * box + box / 2,
                        startCenter[1] + i * box + box / 2,
                    )
                    screen.blit(text, textRect)

        pygame.display.flip()
    pygame.quit()

# Converts a single line of string into array of schema
def strToSchema(s: str):
    return [0 if c == "X" else 1 for c in s.split("\n")[0]]


# checks if a position is valid
def isValid(schema: np.ndarray, i, j):
    return (
        i >= 0
        and i < schema.shape[0]
        and j >= 0
        and j < schema.shape[1]
        and schema[i][j] != 0
    )


# Performs belief update based on a give move
# Note that it creates a new copy of beliefs
def beliefUpdateStep(schema: np.ndarray, beliefs: np.ndarray, move: Move):
    newBeliefs = np.zeros(beliefs.shape)
    for i in range(0, beliefs.shape[0]):
        for j in range(0, beliefs.shape[1]):
            if schema[i][j] != 0:
                # update belief for unblocked cells
                newI = i + move.value[0]
                newJ = j + move.value[1]
                if isValid(schema, newI, newJ):
                    newBeliefs[newI][newJ] += beliefs[i][j]
                else:
                    newBeliefs[i][j] += beliefs[i][j]
    return newBeliefs


def getHeuristic(p: np.ndarray,dists):
    maxLen = 0
    nonZero = []
    for i in range(0, p.shape[0]):
        for j in range(0, p.shape[1]):
            if p[i][j] != 0:
                nonZero.append((i,j))
    
    for x in nonZero:
        for y in nonZero:
            maxLen = max(maxLen,dists[(x,y)])

    return maxLen


def getHeuristicOld(p: np.ndarray):
    maxVal = np.max(p)
    listKeys = []
    for i in range(0, p.shape[0]):
        for j in range(0, p.shape[1]):
            if p[i][j] == maxVal:
                listKeys.append((i, j))
    if len(listKeys) == 1:
        secondMax = np.max(p[p != maxVal])
        listKeysSecond = []
        for i in range(0, p.shape[0]):
            for j in range(0, p.shape[1]):
                if p[i][j] == secondMax:
                    listKeysSecond.append((i, j))
        h = 0
        for i in range(0, len(listKeysSecond)):
            h = max(
                h,
                max(listKeys[0][0], listKeysSecond[i][0])
                + max(listKeys[0][1], listKeysSecond[i][1]),
            )
        return h
    h = 0
    for i in range(0, len(listKeys)):
        for j in range(0, len(listKeys)):
            if i != j:
                h = max(
                    h,
                    max(listKeys[i][0], listKeys[j][0])
                    + max(listKeys[i][1], listKeys[j][1]),
                )
    return h


def simulateGame(schema, moves):
    beliefs = np.array(schema != 0, dtype=int) / np.count_nonzero(schema != 0)
    # print(beliefs)

    for move in moves:
        # print("Move :",move.name," ===========================")
        beliefs = beliefUpdateStep(schema, beliefs, move)
    #     print(beliefs)
    return beliefs


def getBestPath(schema, a, b):
    # Performs search algorithm from top left non-zero to bottom-right non-zero
    fringe = PriorityQueue()
    fringe.put((0, a))

    costs = {}
    pred = {}

    costs[a] = 0
    pred[a] = None

    while not fringe.empty():
        prior, top = fringe.get()
        if top == b:
            break

        for move in Move:
            newCost = costs[top] + 1
            newPoint = (top[0] + move.value[0], top[1] + move.value[1])
            if isValid(schema, newPoint[0], newPoint[1]) and (
                newPoint not in costs or costs[newPoint] > newCost
            ):
                costs[newPoint] = newCost
                pred[newPoint] = top
                priority = newCost + abs(newPoint[0] - b[0]) + abs(newPoint[1] - b[1])
                fringe.put((priority, newPoint))
    
    curr = pred[b]
    length = 0
    while not curr is None:
        curr = pred[curr]
        length +=1

    return length




def getPathCache(schema):
    dists = {}
    
    nonZero = []
    for i in range(0, schema.shape[0]):
        for j in range(0, schema.shape[1]):
            if schema[i][j] != 0:
                nonZero.append((i, j))
    for i in nonZero:
        for j in nonZero:
                # print( nonZero[i], nonZero[j])
                dists[(i, j)] = getBestPath(schema, i,j)
                # print(path)
    return dists

def getRating(beliefs,dists):
    rating = 0.0
    nonZero = []
    for i in range(0, beliefs.shape[0]):
        for j in range(0, beliefs.shape[1]):
            if beliefs[i][j] != 0:
                nonZero.append((i, j))

    for i in range(0, len(nonZero)):
        for j in range(len(nonZero) - 1, -1, -1):
            rating += dists[(nonZero[i], nonZero[j])]

    return rating

# Performs greedy search with random tie-breaking
def getGreedyMoveSequence(schema, ui=False,pathCache = None):
    beliefs = np.array(schema != 0, dtype=int) / np.count_nonzero(schema == 1)

    totalTiles = beliefs.shape[0] * beliefs.shape[1]

    # printStuff("Total tiles: ",totalTiles.)
    bestMovesSequence = []
    if pathCache is None:
        dists = getPathCache(schema)
    else:
        dists=pathCache
    
    moveCounter = {
        move:0 for move in Move
    }

    while np.count_nonzero(beliefs == 0.0) != totalTiles - 1:

        tmpBeliefs = {}
        for move in Move:
            tmpBeliefs[move] = beliefUpdateStep(schema, beliefs, move)

        zeros = [np.count_nonzero(val == 0.0) for val in tmpBeliefs.values()]
        mx = max(zeros)

        keyList = list(tmpBeliefs.keys())
        bestMoves = [keyList[i] for i in range(0, len(zeros)) if zeros[i] == mx]

        if len(bestMoves) > 1:
            contours = {i: getRating(tmpBeliefs[i],dists) for i in bestMoves}
            mxContours = min(contours.values())
            bestMoves = [i for i in contours.keys() if contours[i] == mxContours]
            if len(bestMoves)!=1:
                rank = {x:moveCounter[x] for x in bestMoves}
                mn = min(rank.values())
                for x in bestMoves:
                    if moveCounter[x]==mn:
                        bestMove = x
                        break
            else:
                bestMove = bestMoves[0]
        else:
            bestMove = bestMoves[0]
        
        beliefs = tmpBeliefs[bestMove]
        moveCounter[bestMove]+=1
        bestMovesSequence.append(bestMove)

    return bestMovesSequence


def getBFSMovesSequence(schema):
    global debug
    # initial beliefs
    beliefs = np.array(schema != 0, dtype=int) / np.count_nonzero(schema == 1)

    fringe = deque()
    fringe.append(([], beliefs))

    totalTiles = beliefs.shape[0] * beliefs.shape[1]
    printStuff("Total tiles: ", totalTiles)
    printStuff("============")

    bestPath = None

    dists = getPathCache(schema)

    while len(fringe) != 0:
        seq, beliefs = fringe.popleft()

        printStuff(
            "[Fringe size : %d] [curr len : %d]"
            % (len(fringe), len(seq)),
            end="\r",
        )
        # goal state
        if np.count_nonzero(beliefs == 0.0) == totalTiles - 1:
            # goal state
            if bestPath is None:
                bestPath = seq
            if len(bestPath) < len(seq):
                bestPath = seq
            # printStuff("Found Goal!")
            break

        tmpBeliefs = {}
        zeros = {}
        for move in Move:
            tmpBeliefs[move] = beliefUpdateStep(schema, beliefs, move)
            zeros[move] = np.count_nonzero(tmpBeliefs[move] == 0.0)

        mx = max(zeros.values())

        bestMoves = [move for move in zeros if zeros[move] == mx]
        
        ratings = {i: getRating(tmpBeliefs[i],dists) for i in bestMoves}
        mxRatings = min(ratings.values())
        bestMoves = [i for i in ratings.keys() if ratings[i] == mxRatings]

        
        for move in bestMoves:
            fringe.append((seq + [move], tmpBeliefs[move]))
        
        if debug:
            printStuff("BestMove: ", bestMoves, log=True)
            printStuff("====", log=True)
            printStuff("Fringe", log=True)
            for f in fringe:
                printStuff(f[0], log=True)
            printStuff("============", log=True)

    return bestPath if not bestPath is None else []


def getAStarMovesSequence(schema: np.ndarray):
    global debug

    totalTile = schema.shape[0] * schema.shape[1]

    beliefs = schema / np.count_nonzero(schema == 1)

    fringe = PriorityQueue()
    fringe.put((0, beliefs.tobytes()))

    costs: dict = {}
    costs[beliefs.tobytes()] = 0

    pathStore = {}
    pathStore[beliefs.tobytes()] = None
    def getPath(stat):
        path = []
        current = pathStore[stat]
        while not current is None:
            path.append(current[1])
            current = pathStore[current[0]]
        path.reverse()  # optional
        return path

    dists = getPathCache(schema)
    greedyPath = getGreedyMoveSequence(schema,pathCache= dists)
    

    endState = simulateGame(schema, greedyPath).tobytes()
    costs[endState] = len(greedyPath)


    pruned = 0
    while not fringe.empty():
        pr, curr = fringe.get()
        currArray = np.frombuffer(curr, dtype=beliefs.dtype).reshape(beliefs.shape)
        # printStuff(
        #     "[Fringe size : %d] [bestPath len : %d] [pruned: %d] [currLen: %d] [estimate: %d] "
        #     % (fringe.qsize(), costs[endState], pruned, costs[curr], pr),
        #     end="\r",
        # )

        if costs[curr] > costs[endState]:
            pruned += 1
            continue
        if pr > costs[endState]:
            pruned += 1
            continue

        if np.count_nonzero(currArray == 0.0) == totalTile - 1:
            if endState is None:
                endState = curr
            if costs[endState] > costs[curr]:
                endState = curr
            continue

        tmpBeliefs = {}
        zeros = {}
        for move in Move:
            tmpBeliefs[move] = beliefUpdateStep(schema, currArray, move)
            zeros[move] = np.count_nonzero(tmpBeliefs[move] == 0.0)
        mx = max(zeros.values())
        bestMoves = [move for move in zeros if zeros[move] == mx]
        ratings = {i: getRating(tmpBeliefs[i],dists) for i in bestMoves}
        mxRatings = min(ratings.values())
        bestMoves = [i for i in ratings.keys() if ratings[i] == mxRatings]
        for move in bestMoves:
            probs = tmpBeliefs[move]
            probString = probs.tobytes()
            newCost = costs[curr] + 1
            if probString not in costs or newCost < costs[probString]:
                costs[probString] = newCost
                fringe.put((newCost + getHeuristic(probs,dists), probs.tobytes()))

                pathStore[probString] = (curr, move)
    # print()
    # print(np.frombuffer(endState, dtype=beliefs.dtype).reshape(beliefs.shape))
    if endState not in pathStore:
        # print("Greedy")
        return greedyPath
    return [] if endState is None else getPath(endState)


def getSusMovesSequence(schema: np.ndarray):
    beliefs = schema / np.count_nonzero(schema == 1)

    def getNonZeroBounds(p, lC, rC):
        if lC:
            # left top - right bottom
            minY, maxY = p.shape[1], 0

            minX = 0 if rC != 0 else p.shape[0]
            maxX = 0 if rC == 0 else p.shape[0]
            for i in range(0, p.shape[0]):
                for j in range(0, p.shape[1]):
                    # print(p[i][j], " | ",(i,j))
                    if p[i][j] != 0:
                        minY = min(minY, j)
                        maxY = max(maxY, j)
            for i in range(0, p.shape[0]):
                if p[i][minY] != 0:
                    if rC == 0:
                        minX = min(minX, i)
                    else:
                        minX = max(minX, i)
                if p[i][maxY] != 0:
                    if rC == 1:
                        maxX = min(maxX, i)
                    else:
                        maxX = max(maxX, i)
            return (minX, minY), (maxX, maxY)
        else:
            # top left - bottom right
            # left top - right bottom
            minY, maxY = p.shape[1], 0

            minX = 0 if rC != 0 else p.shape[0]
            maxX = 0 if rC == 0 else p.shape[0]
            for i in range(0, p.shape[0]):
                for j in range(0, p.shape[1]):
                    # print(p[i][j], " | ",(i,j))
                    if p[i][j] != 0:
                        minY = min(minY, i)
                        maxY = max(maxY, i)
            for i in range(0, p.shape[1]):
                if p[minY][i] != 0:
                    if rC == 0:
                        minX = min(minX, i)
                    else:
                        minX = max(minX, i)
                if p[maxY][i] != 0:
                    if rC == 1:
                        maxX = min(maxX, i)
                    else:
                        maxX = max(maxX, i)
            return (minY, minX), (maxY, maxX)

    def getBestMove(schema, a, b):
        # Performs search algorithm from top left non-zero to bottom-right non-zero
        fringe = PriorityQueue()
        fringe.put((0, a))

        costs = {}
        pred = {}

        costs[a] = 0
        pred[a] = None

        while not fringe.empty():
            prior, top = fringe.get()
            # print("=== [%d]======="%prior)
            # print(top)
            if top == b:
                break

            for move in Move:
                newCost = costs[top] + 1
                newPoint = (top[0] + move.value[0], top[1] + move.value[1])
                if isValid(schema, newPoint[0], newPoint[1]) and (
                    newPoint not in costs or costs[newPoint] > newCost
                ):
                    # print("Added ",move.name," || ", newCost + abs(newPoint[0]-b[0]) + abs(newPoint[1]-b[1]))
                    costs[newPoint] = newCost
                    pred[newPoint] = (top, move)
                    priority = (
                        newCost + abs(newPoint[0] - b[0]) + abs(newPoint[1] - b[1])
                    )
                    fringe.put((priority, newPoint))

        if b not in pred:
            print("No path from ", a, " to ", b)
            return []

        curr = pred[b]
        path = []
        while not curr is None:
            path.append(curr[1])
            curr = pred[curr[0]]

        path.reverse()
        # print(path)
        return path[0]

    moves = {}
    done = False
    beliefStates = {}
    for i in [0, 1]:
        for j in [0, 1]:
            moves[(i, j)] = []
            beliefStates[(i, j)] = deepcopy(beliefs)
    while not done:
        for i in [0, 1]:
            for j in [0, 1]:
                a, b = getNonZeroBounds(beliefStates[(i, j)], i, j)
                if a == b:
                    return moves[(i, j)]
                move = getBestMove(schema, a, b)
                # print((i,j)," -> ",move)
                beliefStates[(i, j)] = beliefUpdateStep(
                    schema, beliefStates[(i, j)], move
                )
                moves[(i, j)].append(move)

    return moves[(0, 1)]

def isValidSchema(schema):
    start = (0,0)
    for i in range(0,schema.shape[0]):
        for j in range(0,schema.shape[1]):
            if isValid(schema,i,j):
                start = (i,j)
                break
    
    fringe = [start]
    visited = np.zeros_like(schema)

    while len(fringe)!=0:
        top = fringe.pop()

        visited[top[0],top[1]] = 1

        for move in Move:
            i,j = top[0]+move.value[0], top[1]+move.value[1]
            if isValid(schema,i,j) and visited[i,j]==0:
                fringe.append((i,j))
    
    for i in range(0,schema.shape[0]):
        for j in range(0,schema.shape[1]):
            if isValid(schema,i,j) and visited[i,j]==0:
                return False
    return True

if __name__ == "__main__":
    start = time()
    args = buildArgs()

    debug = args.debug

    schema = []
    if not args.generate:
        with open(args.schema, "r") as f:
            schema = [strToSchema(x) for x in f.readlines()]
    else:
        schema = []
        for i in range(args.rows):
            row = []
            for j in range(args.columns):
                row.append(1 if random.random()>0.6 else 0)
        
        schema = np.array(schema)
        while not isValid(schema):
            schema = []
            for i in range(args.rows):
                row = []
                for j in range(args.columns):
                    row.append(1 if random.random()>0.6 else 0)
            
            schema = np.array(schema)


    schema = np.array(schema)

    nonZero = []
    for i in range(0, schema.shape[0]):
        for j in range(0, schema.shape[1]):
            if schema[i, j] == 1:
                nonZero.append((i, j))

    schema[random.choice(nonZero)] = 2
    
    if args.algo == 0:
        bestMovesSequence = getAStarMovesSequence(schema)
    elif args.algo == 1:
        bestMovesSequence = getSusMovesSequence(schema)
    elif args.algo == 2:
        bestMovesSequence = getGreedyMoveSequence(schema)
    elif args.algo == 3:
        bestMovesSequence = getBFSMovesSequence(schema)

    clear_output(wait=True)
    if args.ui:
        runUI(schema, bestMovesSequence)
    print("Total time: ",time()-start)
    printStuff("Length of Path: ", len(bestMovesSequence))
    printStuff("Best moves: ", [i.name for i in bestMovesSequence])
