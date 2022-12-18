
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

    def getCost(probs, curr):
        zP = np.count_nonzero(probs==0.0)
        zC = np.count_nonzero(curr==0.0)
        # print(zP,"  < ",zC)
        return 1/zP
    def getPath(stat):
        path = []
        current = pathStore[stat]
        while not current is None:
            path.append(current[1])
            current = pathStore[current[0]]
        path.reverse() # optional
        return path

    endState = None

    while not fringe.empty():
        pr,curr = fringe.get()
        currArray = np.frombuffer(curr,dtype=beliefs.dtype).reshape(beliefs.shape)
        # print("------- curr [%0.4f] --------------------"%pr)
        # print(currArray)

        printStuff("[Fringe size : %d] [bestPath len : %d] [Max Prob : %0.4f]"%(fringe.qsize(), 0 if endState is None else len(endState),np.max(currArray)),end="\r")

        if np.count_nonzero(currArray==0.0)==totalTile - 1:
            currPath = getPath(curr)
            print("Goal!")
            if endState is None:
                endState = currPath
                print(endState)
            if len(endState)>len(currPath):
                endState = currPath
                print(endState)

            continue
        
        
        for move in Move:
            # print(move.name)
            # print("----------------------")

            probs = beliefUpdateStep(schema,currArray,move)
            probString = probs.tobytes()
            newCost = costs[curr] + getCost(probs,currArray)
            if probString not in costs or newCost < costs[probString]:
                # print(probs)
                # print("Added ",move.name,"====>",newCost)
                costs[probString] = newCost
                fringe.put((newCost,probs.tobytes()))
                pathStore[probString] = (curr,move)
    
    return [] if endState is None else endState

def getBestMovesSequence(schema : np.ndarray):
    # define a more stronger goal

    # Runs one iteration of Search to make "cell" 1.0
    def getBestSequenceWithAGoal(schema,cell):
        global debug
        totalTile = schema.shape[0]*schema.shape[1]

        beliefs = schema / np.count_nonzero(schema==1)

        fringe = PriorityQueue()
        fringe.put((0,beliefs.tobytes()))

        costs : dict = {}
        costs[beliefs.tobytes()] = 0
        
        pathStore = {}
        pathStore[beliefs.tobytes()] = None

        def getCost(probs, curr,cell):
            
            # print(probs[cell[0]])
            # print(curr[cell[0]])
            # print("Checking cell: ",cell," : ",probs[cell[0]][cell[1]]," || ",curr[cell[0]][cell[1]])
            val = probs[cell[0]][cell[1]]-curr[cell[0]][cell[1]]
            if val<0:
                return 9999
            elif val==0:
                return 8888
            return 1/val

        def getHeuristic(probs):
            return abs(1-probs[cell[0]][cell[1]])

        def getPath(stat):
            path = []
            current = pathStore[stat]
            while not current is None:
                path.append(current[1])
                current = pathStore[current[0]]
            path.reverse() # optional
            return path

        endState = None

        # while not fringe.empty():
        while fringe.qsize()!=0:
            _,curr = fringe.get()
            currArray = np.frombuffer(curr,dtype=beliefs.dtype).reshape(beliefs.shape)
            printStuff("[Fringe size : %d] [cell state : %d]"%(fringe.qsize(), currArray[cell[0]][cell[1]]),end="\r")
            # print("=====curr=========")
            # print(cell, " ------> ",currArray[cell[0]][cell[1]])
            # print(currArray)
            if abs(currArray[cell[0]][cell[1]]-1) < 1e-4:
                currPath = getPath(curr)
                if endState is None:
                    endState = currPath
                    # print(endState)
                if len(endState)>len(currPath):
                    endState = currPath
                    # print(endState)

                break
            
            
            for move in Move:
                # print(move.name)
                probs = beliefUpdateStep(schema,currArray,move)
                probString = probs.tobytes()
                # print("-----------------")

                newCost = costs[curr] +  getCost(probs,currArray,cell)
                if probString not in costs or newCost < costs[probString]:
                    # print(probs)
                    costs[probString] = newCost 
                    # print("Added ",move.name," --> ",costs[probString])

                    fringe.put((newCost+ getHeuristic(probs),probs.tobytes()))
                    pathStore[probString] = (curr,move)
            # print("=================")
        
        return [] if endState is None else endState

    exploreOrder= {}
    for i in range(schema.shape[0]):
        for j in range(schema.shape[1]):
            if schema[i][j]:
                count = 0
                if isValid(schema,i+1,j):
                    count+=1
                if isValid(schema,i-1,j):
                    count+=1
                if isValid(schema,i,j+1):
                    count+=1
                if isValid(schema,i,j-1):
                    count+=1
                
                exploreOrder[(i,j)] = count
    
    cellsInOrder = list(dict(sorted(exploreOrder.items(),key=lambda item: item[1])).keys())
    bestPath = None
    for cell in cellsInOrder:
        print("Trying: ",cell)  
        tmp = getBestSequenceWithAGoal(schema,cell)
        if bestPath is None or((not bestPath is None) and len(tmp)<len(bestPath)):
            bestPath = tmp
    return bestPath