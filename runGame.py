import pygame
import argparse
from main import strToSchema,Move, beliefUpdateStep
import numpy as np
import math

pygame.init()

def buildArgs():
    parser = argparse.ArgumentParser(description="Script to save the Nuclear power station")
    
    parser.add_argument("--schema",required=False,default="./Thor23-SA74-VERW-Schematic (Classified).txt",help="Path of the schema file")
    return parser.parse_args()


screen = pygame.display.set_mode([900, 900])

running = True


args = buildArgs()
schema = []
with open(args.schema,"r") as f:
    schema = [strToSchema(x) for x in f.readlines()]

schema = np.array(schema)

beliefs = schema / np.count_nonzero(schema==1)

colors = {
    "WHITE":(255,255,255),
    "BLACK":(0,0,0),
}

boardLimits = [800,800]
movesSoFar = []

box = 10
startCenter = [50,50]
if schema.shape[0]>schema.shape[1]:
    box = boardLimits[0]//schema.shape[0]
    startCenter[0] += box*(schema.shape[0]-schema.shape[1])//2
else:
    box = boardLimits[1]//schema.shape[1]
    startCenter[1] += box*(schema.shape[1]-schema.shape[0])//2

font = pygame.font.Font('freesansbold.ttf', math.floor(box*0.35))
totalTiles= beliefs.shape[0] * beliefs.shape[1]


while running:
    if np.count_nonzero(beliefs==0.0)== totalTiles-1:
        running = False
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running = False
        if event.type==pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                movesSoFar.append(Move.LEFT)
                beliefs = beliefUpdateStep(schema, beliefs, movesSoFar[-1])
            if event.key == pygame.K_RIGHT:
                movesSoFar.append(Move.RIGHT)
                beliefs = beliefUpdateStep(schema, beliefs, movesSoFar[-1])
            if event.key == pygame.K_UP:
                movesSoFar.append(Move.UP)
                beliefs = beliefUpdateStep(schema, beliefs, movesSoFar[-1])
            if event.key == pygame.K_DOWN:
                movesSoFar.append(Move.DOWN)
                beliefs = beliefUpdateStep(schema, beliefs, movesSoFar[-1])
            # print(movesSoFar[-1])

    screen.fill(colors["WHITE"])

    pygame.draw.rect(screen,colors["BLACK"],(startCenter[0],startCenter[1],schema.shape[1]*box,schema.shape[0]*box),width=2)

    for i in range(0,schema.shape[0]):
        for j in range(0,schema.shape[1]):
            if not schema[i][j]:
                pygame.draw.rect(screen,colors["BLACK"],(startCenter[0]+j*box,startCenter[1]+i*box,box,box))

    for i in range(0,schema.shape[0]):
        for j in range(0,schema.shape[1]):
            if schema[i][j]:
                text = font.render(("%0.3f"%(beliefs[i][j])),False,colors["BLACK"])
                textRect = text.get_rect()
                textRect.center = (startCenter[0]+j*box+box/2,startCenter[1]+i*box+box/2  )
                screen.blit(text, textRect)

    pygame.display.flip()

pygame.quit()

print("Moves: ",[i.name for i in movesSoFar])
print("Move count: ", len(movesSoFar))