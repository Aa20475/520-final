
def getContour(schema,beliefs):
    minY, maxY= beliefs.shape[1],0
    minX, maxX= beliefs.shape[0],0
    for i in range(0,beliefs.shape[0]):
        for j in range(0,beliefs.shape[1]):
            # print(p[i][j], " | ",(i,j))
            if beliefs[i][j]!=0:
                minY=  min(minY,j)
                maxY= max(maxY,j)
                minX=  min(minX,i)
                maxX= max(maxX,i)
    # print(minX,minY, maxX,maxY)
    points = set()

    for i in range(0,beliefs.shape[0]):
        if beliefs[i][minY]:
            points.add((i,minY))
        if beliefs[i][maxY]:
            points.add((i,maxY))
    for j in range(0,beliefs.shape[1]):
        if beliefs[minX][j]:
            points.add((minX,j))
        if beliefs[maxX][j]:
            points.add((maxX,j))
    
    for i in range(0,beliefs.shape[0]):
        first = False
        x = -1
        for j in range(0,beliefs.shape[1]):
            # print(p[i][j], " | ",(i,j))
            if beliefs[i][j]!=0:
                if not first:
                    points.add((i,j))
                    first= True
                x = max(x,j)
        if x!=-1:
            points.add((i,x))
    
    for i in range(0,beliefs.shape[0]):
        log = ""
        for j in range(0,beliefs.shape[1]):
            if (i,j) in points:
                log+="_"
            else:
                log+="X"
        print(log)
    return list(points)

def getShortestPath(schema,a,b):
    fringe = PriorityQueue()
    fringe.put((0,a))
    
    costs = {}
    pred = {}

    costs[a] = 0
    pred[a] = None
    print(schema)
    
    while not fringe.empty():
        prior,top = fringe.get()
        print("=== [%d]======="%prior)
        print(top)
        if top == b:
            break
        
        for move in Move:
            newCost = costs[top]+1
            newPoint = (top[0]+move.value[0],top[1]+move.value[1])
            if isValid(schema,newPoint[0],newPoint[1]) and (newPoint not in costs or costs[newPoint]>newCost):
                print("Added ",move.name," || ", newCost + abs(newPoint[0]-b[0]) + abs(newPoint[1]-b[1]))
                costs[newPoint] = newCost
                pred[newPoint] = (top, move)
                priority = newCost + abs(newPoint[0]-b[0]) + abs(newPoint[1]-b[1])
                fringe.put((priority,newPoint))

    if b not in pred:
        print("No path from ",a," to ",b)
        return []
    
    curr = pred[b]
    moves = []
    path = [b]
    while not curr is None:
        moves.append(curr[1])
        path.append(curr[0])
        curr = pred[curr[0]]
    
    path.reverse()
    moves.reverse()
    # print(path)
    return moves,path

def getBestMovesList(schema,beliefs):
    counter = {}
    for move in Move:
        tmpBeliefs = beliefUpdateStep(schema,beliefs,move)
        counter[move] = getDistanceBetweenContours(schema,tmpBeliefs)
    maxVal = max(counter.values())

    return [k for k in counter.keys() if counter[k]==maxVal]

def getDistanceBetweenContours(schema,beliefs):
    # Performs search algorithm from top left non-zero to bottom-right non-zero
    contour = getContour(schema,beliefs)
    
    distBetweenContours = 0
    for i in range(0,len(contour)):
        for j in range(0,len(contour)):
            if contour[i]!=contour[j]:
                print(contour[i]," :: ",contour[j])
                moves , _ = getShortestPath(schema,contour[i],contour[j])
                distBetweenContours += len(moves)

    return distBetweenContours