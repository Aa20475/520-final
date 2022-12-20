from main import getSusMovesSequence, buildArgs, isValid, Move
from queue import PriorityQueue
import numpy as np
from tqdm import tqdm

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

def getSimilarSchemas(schema):
    rotStrings = []
    for _ in range(0,3):
        rotStrings.append(np.rot90(schema,1).tobytes())

    hFlip = np.flip(schema,0)
    rotStrings.append(hFlip.tobytes())
    for _ in range(0,3):
        rotStrings.append(np.rot90(hFlip,1).tobytes())
    
    vFlip = np.flip(schema,1)
    rotStrings.append(vFlip.tobytes())
    for _ in range(0,3):
        rotStrings.append(np.rot90(vFlip,1).tobytes())
    
    return rotStrings

def getBFSWorstSchema(size):
    # BFS with aggressive pruning
    fringe = PriorityQueue()
    start =  np.ones(size)
    startSteps = len(getSusMovesSequence(start))
    fringe.put((1/startSteps,start.tobytes()))

    mxSchema = start
    mxSteps = startSteps

    visited = set()
    pruned = 0
    flagPrune = 0

    while fringe.qsize()!=0:
        priority , currBytes = fringe.get()
        curr = np.frombuffer(currBytes,start.dtype).reshape(size)
        print(
            "[Fringe size : %d] [bestSchema len : %d] [currLen: %d] [pruned: %d] [flagPrune: %d]"
            % (fringe.qsize(), mxSteps, 1/priority, pruned, flagPrune),
            end="\r",
        )

        visited.add(currBytes)

        if mxSteps < 1/priority:
            mxSchema = curr
            mxSteps = 1/priority

        validNextSchemas = []
        
        for i in range(0,size[0]):
            for j in range(0,size[1]):
                if isValid(curr,i,j):
                    nextSchema = np.array(curr)
                    nextSchema[i,j] = 0
                    if isValidSchema(nextSchema):
                        # If it is connected
                        validNextSchemas.append(nextSchema)
                    else:
                        pruned +=1
        
        unique = [True]*len(validNextSchemas)
        for i in range(0,len(unique)):
            if unique[i]:
                rotStrings = getSimilarSchemas(validNextSchemas[i])
                for j in range(i+1,len(unique)):
                    if validNextSchemas[j].tobytes() in rotStrings:
                        flagPrune+=1
                        unique[j] = False
        
        for schema in validNextSchemas:
            if unique:
                fringe.put((1/len(getSusMovesSequence(schema)),schema.tobytes()))
                

    
    return mxSchema


def getWorstSchema(size):
    schema = np.ones(size)
    
    while True:
        actionToSequence = {}
        print(len(getSusMovesSequence(schema)))
        for i in tqdm(range(0,size[0]),leave=False):
            log = ""
            for j in tqdm(range(0,size[1]),leave=False):
                if isValid(schema,i,j):
                    tmpSchema = np.array(schema)
                    tmpSchema[i,j] = 0
                    actionToSequence[(i,j)] = len(getSusMovesSequence(tmpSchema))
                    log += str(actionToSequence[(i,j)]) + " "
            # print(log)

        
        print(max(actionToSequence.values()))


    return "Brrr"


if __name__ == "__main__":

    args = buildArgs()

    size = (args.rows,args.columns)
    print("Exploring %d X %d graph space!"%(size[0],size[1]))


    print(getWorstSchema(size))