import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import pygame

def draw_arrow(
        surface: pygame.Surface,
        start: pygame.Vector2,
        end: pygame.Vector2,
        color: pygame.Color,
        body_width: int = 2,
        head_width: int = 4,
        head_height: int = 2,
    ):
    """Draw an arrow between start and end with the arrow head at the end.

    Args:
        surface (pygame.Surface): The surface to draw on
        start (pygame.Vector2): Start position
        end (pygame.Vector2): End position
        color (pygame.Color): Color of the arrow
        body_width (int, optional): Defaults to 2.
        head_width (int, optional): Defaults to 4.
        head_height (float, optional): Defaults to 2.
    """
    arrow = start - end
    angle = arrow.angle_to(pygame.Vector2(0, -1))
    body_length = arrow.length() - head_height

    # Create the triangle head around the origin
    head_verts = [
        pygame.Vector2(0, head_height / 2),  # Center
        pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
        pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
    ]
    # Rotate and translate the head into place
    translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(-angle)
    for i in range(len(head_verts)):
        head_verts[i].rotate_ip(-angle)
        head_verts[i] += translation
        head_verts[i] += start

    pygame.draw.polygon(surface, color, head_verts)

    # Stop weird shapes when the arrow is shorter than arrow head
    if arrow.length() >= head_height:
        # Calculate the body rect, rotate and translate into place
        body_verts = [
            pygame.Vector2(-body_width / 2, body_length / 2),  # Topleft
            pygame.Vector2(body_width / 2, body_length / 2),  # Topright
            pygame.Vector2(body_width / 2, -body_length / 2),  # Bottomright
            pygame.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
        ]
        translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
        for i in range(len(body_verts)):
            body_verts[i].rotate_ip(-angle)
            body_verts[i] += translation
            body_verts[i] += start

        pygame.draw.polygon(surface, color, body_verts)


POINTS = {
    "P": -1000,
    "R": 100,
    "S": 0,
    ".": 0
}
        
        
class State:
    
    def __init__(self):
        self.value = None
        self.tag = None
    
    def setTag(self,tag):
        self.tag = tag
        self.val = POINTS[tag]
            
        
      
class Model:
    
    def __init__(self):
        self.grids = None
        self.dims = None
        self.folder = None
        
        self.nrStates = None
        self.nrActions = None
        self.states = None
        self.actions = None
        
        self.moves = [[0,1],[0,-1],[1,0],[-1,0]]
        
    def __setup__(self):
        self.dims, self.grids = self.readFiles(self.folder)
        self.nrStates = self.dims[0]*self.dims[1]*self.dims[2]
        
        self.__flattenStates__()
        self.__defineActions__()
        
    def __flattenStates__(self):
        self.states = [0]*self.nrStates
        c = 0
        
        for grid in range(self.dims[2]):
            for row in range(self.dims[1]):
                for col in range(self.dims[0]):
                    self.states[c] = State()
                    self.states[c].setTag(self.grids[grid][row][col])
                    c += 1
              
    def __defineActions__(self):
        normalState = []
        self.actions = dict()
        self.nrActions = 0
        
        for i in range(self.dims[1]):
            normalState.append([])
            for j in range(self.dims[0]):
                normalState[i].append(i*self.dims[1] + j)
        
        for row in range(self.dims[1]):
            for col in range(self.dims[0]):
                
                # Check for moves for each grid cell
                pos = []     

                
                for move in self.moves:
                    # Move is within grid
                    if 0 <= col+move[0] < self.dims[0] and 0 <= row+move[1] < self.dims[1]:
                        pos.append(normalState[row+move[1]][col+move[0]])
                    #else: #<- move to itself
                    #    if normalState[row][col] not in pos:
                    #        pos.append(normalState[row][col])
                
                # Copy found moves for each dimension (gridsize is static)
                for grid in range(self.dims[2]):
                    self.actions[normalState[row][col]+(grid*9)] = sorted([x+(((grid+1)%self.dims[2])*9) for x in pos])
                    self.nrActions += len(pos)
        
    def useFolder(self,folder):
        self.folder = folder

    def readFiles(self,folder):
        grids = []
        dims = None
        
        with open(f"{folder}//0.txt","r") as f:
            dims = [int(x) for x in f.readline().split()]
            
            grids.append([[x for x in line.rstrip()] for line in f.readlines()])
            
        c = 1
        done = False
        while not done:
            try:
                with open(f"{folder}//{c}.txt") as f:
                    grids.append([[x for x in line.rstrip()] for line in f.readlines()])
                c += 1
            except:
                done = True
        
        return dims+[c], grids
                
    def draw2D(self):
        
        # Define some colors
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        
        pygame.init()

        size = (700, 500)
        screen = pygame.display.set_mode(size)

        done = False

        W,M = 30, 40
        
        clock = pygame.time.Clock()
        

      
        # -------- Main Program Loop -----------
        while not done:

            
            screen.fill(BLACK)
            for i in range(self.dims[2]):
                for j in range(self.dims[0]*self.dims[1]):
                    # Draw state
                    if self.states[i*self.dims[0]*self.dims[1] + j].tag == "S":
                        pygame.draw.rect(screen, (255,255,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    elif self.states[i*self.dims[0]*self.dims[1] + j].tag == "P":
                        pygame.draw.rect(screen, (255,0,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    elif self.states[i*self.dims[0]*self.dims[1] + j].tag == "R":
                        pygame.draw.rect(screen, (0,255,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    else:
                        pygame.draw.rect(screen, (255,255,255), [j*(W+M)+M,i*(W+M)+M,W,W]) 
            
            for i in range(self.dims[2]-1):
                for j in range(self.dims[0]*self.dims[1]):        
                    # Draw actions
                    for action in self.actions[i*self.dims[0]*self.dims[1] + j]:
                        draw_arrow(screen, pygame.Vector2(j*(W+M)+M + W//2,i*(W+M)+M+ W//2),pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,action//(self.dims[0]*self.dims[1])*(W+M)+M+ W//2)),(0,255,255),head_width=16,head_height=8,body_width=1)
            
            
            # Last layer actions
            i = self.dims[2]-1
            for j in range(self.dims[0]*self.dims[1]):        
                # Draw actions
                for action in self.actions[i*self.dims[0]*self.dims[1] + j]:
                    # Last layer fade
                    draw_arrow(screen, pygame.Vector2(j*(W+M)+M + W//2,i*(W+M)+M+ W//2),pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(i+1)*(W+M)+M+ W//2)),(0,255,255),head_width=16,head_height=8,body_width=1)

            # Last layer actions
            i = -1
            for j in range(self.dims[0]*self.dims[1]):        
                # Draw actions
                for action in self.actions[(i+1)*self.dims[0]*self.dims[1] + j]:
                    # First layer fade
                    draw_arrow(screen, pygame.Vector2(j*(W+M)+M + W//2,i*(W+M)+M+ W//2),pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(i+1)*(W+M)+M+ W//2)),(0,255,255),head_width=16,head_height=8,body_width=1)

             
                    
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            clock.tick(60)
        
        # Close the window and quit.
        pygame.quit()
        
        
        
    def summary(self):
        print(f"Number of states: {self.nrStates}")
        print(f"Number of actions: {self.nrActions}")
        print(f"States: {self.states}")
        print(f"Actions: {self.actions}")

def main():
    model = Model()
    model.useFolder("grids")
    model.__setup__()
    
    model.summary()
    
    model.draw2D()



if __name__ == "__main__":
    main()