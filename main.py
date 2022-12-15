import numpy as np
import pygame
from pygame.locals import *

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
    "P": 0,
    "R": 100,
    "S": 0,
    ".": 0,
    "X": 0
}
        
        
class State:
    
    def __init__(self):
        self.value = None
        self.tag = None
    
    def setTag(self,tag):
        self.tag = tag
        self.value = POINTS[tag]
    
class Action:

    def __init__(self, stateIndex):
        self.endStateIndex = stateIndex


    
      
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
        self.path = None
        
    def __setup__(self,moveToItself=False):

        def __flattenStates__():
            self.states = [0]*self.nrStates
            c = 0
            
            for grid in range(self.dims[2]):
                for row in range(self.dims[1]):
                    for col in range(self.dims[0]):
                        self.states[c] = State()
                        self.states[c].setTag(self.grids[grid][row][col])
                        c += 1
                
        def __defineActions__():
            normalState = []
            self.actions = dict()
            self.nrActions = 0

            for i in range(self.dims[1]):
                normalState.append([])
                for j in range(self.dims[0]):
                    normalState[i].append(i*self.dims[0] + j)

            for row in range(self.dims[1]):
                for col in range(self.dims[0]):
                    
                    # Check for moves for each grid cell
                    pos = []     

                    
                    for move in self.moves:
                        # Move is within grid
                        if 0 <= col+move[0] < self.dims[0] and 0 <= row+move[1] < self.dims[1]:
                            pos.append(normalState[row+move[1]][col+move[0]])
                        else: #<- move to itself
                            if moveToItself and normalState[row][col] not in pos:
                                pos.append(normalState[row][col])
                    
                    # Copy found moves for each dimension (gridsize is static)
                    for grid in range(self.dims[2]):
                        
                        # Check if cell is not marked with X:
                        if self.states[normalState[row][col]+(grid*(self.dims[0]*self.dims[1]))].tag != "X":
                            actions = [x+(((grid+1)%self.dims[2])*(self.dims[0]*self.dims[1])) for x in pos]
                            actions = [x for x in actions if self.states[x].tag != 'X']  # Filters out blocked states
                            actions = [Action(x) for x in actions]          # Makes action classes
                            
                            self.actions[normalState[row][col]+(grid*(self.dims[0]*self.dims[1]))] = actions
                            self.nrActions += len(pos)
                        else:
                            self.actions[normalState[row][col]+(grid*(self.dims[0]*self.dims[1]))] = []
    
        self.dims, self.grids = self.readFiles(self.folder)
        self.nrStates = self.dims[0]*self.dims[1]*self.dims[2]
        
        __flattenStates__()
        __defineActions__()

    def assignGridValues(self):
        """
        Dynamically assigns values to states
        """
        change = True
        learningRate = 0.9

        iterations = 0
        
        while change:
            change = False

            for state in range(self.nrStates):
                for action in self.actions[state]:
                    newValue = self.states[action.endStateIndex].value * learningRate
                    if self.states[state].value < newValue and self.states[state].tag != "P":
                        self.states[state].value = newValue
                        change = True
            
            iterations += 1

        print(f"Took {iterations} iterations.")

    def greedyPolicy(self):
        self.path = []

        # Find starting state
        for i in range(self.nrStates):
            if self.states[i].tag == "S":
                self.path.append(i)
        
        # While last state is no reward state
        while self.states[self.path[-1]].tag != "R":
            next = None
            nextval = -1000
            
            for action in self.actions[self.path[-1]]:
                if self.states[action.endStateIndex].value > nextval:
                    nextval = self.states[action.endStateIndex].value
                    next = action.endStateIndex
                
            self.path.append(next)
        
        print(f"Optimal greedy path: {self.path}")

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
                
    def drawModel(self):
        """
        Draws the state-transition diagram
        Could be gigantic, beware!
        If a path is generated (for example greedyPolicy), it will draw
        the arrows of transition in gold
        """
        # Define some colors
        BLACK = (0, 0, 0)
        W,M = 30,10

        pygame.init()

        size = ((W+M)*self.dims[0]*self.dims[1]+M, (W+M)*self.dims[2]+M)
        screen = pygame.display.set_mode(size)

        mouseHover = True           # To show actions of state you are currently hovering over

        done = False
        clock = pygame.time.Clock()
        
        def drawStates():
            """
            Draws all states in a self.dims[0]*self.dims[1] by self.dims[2] grid
            """
            for i in range(self.dims[2]):
                for j in range(self.dims[0]*self.dims[1]):
                    # Draw state
                    if self.states[i*self.dims[0]*self.dims[1] + j].tag == "S":
                        pygame.draw.rect(screen, (255,255,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    elif self.states[i*self.dims[0]*self.dims[1] + j].tag == "P":
                        pygame.draw.rect(screen, (255,0,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    elif self.states[i*self.dims[0]*self.dims[1] + j].tag == "R":
                        pygame.draw.rect(screen, (0,255,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    elif self.states[i*self.dims[0]*self.dims[1] + j].tag == "X":
                        pass
                    else:
                        pygame.draw.rect(screen,(255,255,255), [j*(W+M)+M,i*(W+M)+M,W,W])

        def drawActions(markedArrows=[],gradient=True,drawFromPunishment=False,drawFromReward=False):
            """
            Draws for each state all actions. The last layer goes over to the first layer (gridwise)
            :param markedArrows: list actions in consecutive order (default = [])
            """
            for i in range(self.dims[2]-1):
                for j in range(self.dims[0]*self.dims[1]):        
                    # Draw actions
                    for action in self.actions[i*self.dims[0]*self.dims[1] + j]:
                        # Path arrows color
                        if (i*self.dims[0]*self.dims[1] + j, action.endStateIndex) in markedArrows:
                            color = (255,255,0)
                            bwidth = 4
                        else:
                            color = Color('lightgray')
                            if gradient: # for colors floodfill
                                score = self.states[i*self.dims[0]*self.dims[1] + j].value / POINTS['R']
                                if self.states[i*self.dims[0]*self.dims[1] + j].tag != 'P':
                                    color = (255 - (255 * score), 255 * score, 255 - (255 * score))
                            bwidth = 1
                        
                        # Draw all arrows, even from punishment?
                        if self.states[i*self.dims[0]*self.dims[1] + j].tag != 'P' or drawFromPunishment:
                            if self.states[i*self.dims[0]*self.dims[1] + j].tag != 'R' or drawFromReward:
                                draw_arrow(screen, pygame.Vector2(j*(W+M)+M + W//2,i*(W+M)+M+ W//2),
                                    pygame.Vector2((action.endStateIndex%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,action.endStateIndex//(self.dims[0]*self.dims[1])*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
            
            
            # First/Last layer actions
            i = self.dims[2]-1
            for j in range(self.dims[0]*self.dims[1]):        
                # Draw actions
                for action in self.actions[i*self.dims[0]*self.dims[1] + j]:
                    if (i*self.dims[0]*self.dims[1] + j, action.endStateIndex) in markedArrows:
                        color = (255,255,0)
                        bwidth = 4
                    else:
                        color = Color('lightgray')
                        if gradient: # for colors floodfill
                            score = self.states[i*self.dims[0]*self.dims[1] + j].value / POINTS['R']
                            if self.states[i*self.dims[0]*self.dims[1] + j].tag != 'P':
                                color = (255 - (255 * score), 255 * score, 255 - (255 * score))
                        bwidth = 1
                    # Last layer fade
                    # Draw all arrows, even from punishment?
                    if self.states[i*self.dims[0]*self.dims[1] + j].tag != 'P' or drawFromPunishment:
                        if self.states[i*self.dims[0]*self.dims[1] + j].tag != 'R' or drawFromReward:
                            draw_arrow(screen, pygame.Vector2(j*(W+M)+M + W//2,i*(W+M)+M+ W//2),
                                pygame.Vector2((action.endStateIndex%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(i+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                            draw_arrow(screen, pygame.Vector2(j*(W+M)+M + W//2,-1*(W+M)+M+ W//2),
                                pygame.Vector2((action.endStateIndex%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(-1+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)

        def hoverMouseActions():
            """
            Hover your mouse over a state to see what actions you can take
            """
            mposx, mposy = pygame.mouse.get_pos()
            stateindx, stateindy = mposx // (W+M), mposy // (W+M)
            stateindx, stateindy = min((self.dims[0])*(self.dims[1])-1,stateindx), min(self.dims[2]-1,stateindy)
            actions = self.actions[stateindy*self.dims[0]*self.dims[1] + stateindx]
            for action in actions:
                color = Color('blue')
                bwidth = 4
                
                if stateindy != self.dims[2]-1:
                    draw_arrow(screen, pygame.Vector2(stateindx*(M+W)+M + W//2,stateindy*(M+W)+M + W//2),
                            pygame.Vector2((action.endStateIndex%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,action.endStateIndex//(self.dims[0]*self.dims[1])*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                else: # top bottom
                    draw_arrow(screen, pygame.Vector2(stateindx*(W+M)+M + W//2,stateindy*(W+M)+M+ W//2),
                            pygame.Vector2((action.endStateIndex%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(stateindy+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                    draw_arrow(screen, pygame.Vector2(stateindx*(W+M)+M + W//2,-1*(W+M)+M+ W//2),
                            pygame.Vector2((action.endStateIndex%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(-1+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)


        # -------- Main Program Loop -----------
        while not done:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        mouseHover = not mouseHover
            
            screen.fill(BLACK)
            
            # Draws all states
            drawStates()
            
            # Mark arrows if there is a path
            if self.path != None:
                drawActions([(self.path[i],self.path[i+1]) for i in range(len(self.path)-1)],gradient=False)
            else:
                drawActions()

            # Mouse to show actions
            if mouseHover: hoverMouseActions()

            # Blit to screen
            pygame.display.update()

            clock.tick(60)
        
        # Close the window and quit.
        pygame.quit()
    
    def drawAnimation(self,path=None):
        
        # Define some colors
        BLACK = (0, 0, 0)
        W,M = 30, 10

        pygame.init()

        size = ((W+M)*self.dims[0]+M, (W+M)*self.dims[1]+M)
        screen = pygame.display.set_mode(size)

        c = 0

        done = False
        clock = pygame.time.Clock()
        autoAnimation = True

        def drawStates(grid):
            """
            Draws all the states of the current layer
            """
            for i in range(self.dims[1]):
                for j in range(self.dims[0]):
                    if (grid[i][j] == "S" and self.path == None) or (c%self.dims[2])*self.dims[0]*self.dims[1] + i*self.dims[0] + j == self.path[c%len(self.path)]:
                        pygame.draw.rect(screen, (255,255,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    elif grid[i][j] == "P":
                        pygame.draw.rect(screen, (255,0,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    elif grid[i][j] == "R":
                        pygame.draw.rect(screen, (0,255,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    elif grid[i][j] == "X":
                        pass
                    else:
                        pygame.draw.rect(screen, (255,255,255), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                
        # -------- Main Program Loop -----------
        while not done:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        autoAnimation = not autoAnimation
                    elif event.key == pygame.K_LEFT:
                        autoAnimation = False
                        c -= 1
                    elif event.key == pygame.K_RIGHT:
                        autoAnimation = False
                        c += 1
            
            screen.fill(BLACK)
            
            grid = self.grids[c%self.dims[2]]

            drawStates(grid)
                    
            pygame.display.update()

            clock.tick(2)
            if autoAnimation: c += 1
            
            # reset animation if path exists
            if path != None:
                c = (c+len(path))%len(path)

        # Close the window and quit.
        pygame.quit()
    
    def drawFull(self,path=None):
        """
        Draws the state-transition diagram
        Could be gigantic, beware!
        If a path is generated (for example greedyPolicy), it will draw
        the arrows of transition in gold
        """
        # Define some colors
        BLACK = (0, 0, 0)
        W,M = 30,10

        pygame.init()

        size = ((W+M)*self.dims[0]*self.dims[1]+M, (W+M)*self.dims[2]+M + (W+M)*self.dims[1]+M+W)
        screen = pygame.display.set_mode(size)

        mouseHover = False           # To show actions of state you are currently hovering over

        done = False
        clock = pygame.time.Clock()
        autoAnimation = True
        c = 0
        
        def drawStatesAnimation(grid):
            """
            Draws all the states of the current layer
            """
            exHeight = (W+M)*self.dims[2]+M + W
            exWidth = ((W+M)*self.dims[0]*self.dims[1]+M)/2 - ((W+M)*self.dims[0]+M)/2
            for i in range(self.dims[1]):
                for j in range(self.dims[0]):
                    if (grid[i][j] == "S" and self.path == None) or (c%self.dims[2])*self.dims[0]*self.dims[1] + i*self.dims[0] + j == self.path[c%len(self.path)]:
                        pygame.draw.rect(screen, (255,255,0), [j*(W+M)+M+exWidth,i*(W+M)+M+exHeight,W,W]) 
                    elif grid[i][j] == "P":
                        pygame.draw.rect(screen, (255,0,0), [j*(W+M)+M+exWidth,i*(W+M)+M+exHeight,W,W]) 
                    elif grid[i][j] == "R":
                        pygame.draw.rect(screen, (0,255,0), [j*(W+M)+M+exWidth,i*(W+M)+M+exHeight,W,W]) 
                    elif grid[i][j] == "X":
                        pass
                    else:
                        pygame.draw.rect(screen, (255,255,255), [j*(W+M)+M+exWidth,i*(W+M)+M+exHeight,W,W]) 
                
        def drawStates():
            """
            Draws all states in a self.dims[0]*self.dims[1] by self.dims[2] grid
            """
            for i in range(self.dims[2]):
                for j in range(self.dims[0]*self.dims[1]):
                    # Draw state
                    if self.states[i*self.dims[0]*self.dims[1] + j].tag == "S":
                        pygame.draw.rect(screen, (255,255,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    elif self.states[i*self.dims[0]*self.dims[1] + j].tag == "P":
                        pygame.draw.rect(screen, (255,0,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    elif self.states[i*self.dims[0]*self.dims[1] + j].tag == "R":
                        pygame.draw.rect(screen, (0,255,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    elif self.states[i*self.dims[0]*self.dims[1] + j].tag == "X":
                        pass
                    else:
                        pygame.draw.rect(screen,(255,255,255), [j*(W+M)+M,i*(W+M)+M,W,W])

        def drawActions(markedArrows=[],gradient=True,drawFromPunishment=False,drawFromReward=False,currentAction=None):
            """
            Draws for each state all actions. The last layer goes over to the first layer (gridwise)
            :param markedArrows: list actions in consecutive order (default = [])
            """
            for i in range(self.dims[2]-1):
                for j in range(self.dims[0]*self.dims[1]):        
                    # Draw actions
                    for action in self.actions[i*self.dims[0]*self.dims[1] + j]:
                        # Path arrows color
                        if (i*self.dims[0]*self.dims[1] + j, action.endStateIndex) in markedArrows:
                            color = (255,255,0)
                            bwidth = 4
                            if (i*self.dims[0]*self.dims[1] + j, action.endStateIndex) == currentAction:
                                color = (0,255,0) # Current animation path
                        else:
                            color = Color('lightgray')
                            if gradient: # for colors floodfill
                                score = self.states[i*self.dims[0]*self.dims[1] + j].value / POINTS['R']
                                if self.states[i*self.dims[0]*self.dims[1] + j].tag != 'P':
                                    color = (255 - (255 * score), 255 * score, 255 - (255 * score))
                            bwidth = 1
                        
                        # Draw all arrows, even from punishment?
                        if self.states[i*self.dims[0]*self.dims[1] + j].tag != 'P' or drawFromPunishment:
                            if self.states[i*self.dims[0]*self.dims[1] + j].tag != 'R' or drawFromReward:
                                draw_arrow(screen, pygame.Vector2(j*(W+M)+M + W//2,i*(W+M)+M+ W//2),
                                    pygame.Vector2((action.endStateIndex%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,action.endStateIndex//(self.dims[0]*self.dims[1])*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
            
            
            # First/Last layer actions
            i = self.dims[2]-1
            for j in range(self.dims[0]*self.dims[1]):        
                # Draw actions
                for action in self.actions[i*self.dims[0]*self.dims[1] + j]:
                    if (i*self.dims[0]*self.dims[1] + j, action.endStateIndex) in markedArrows:
                        color = (255,255,0)
                        bwidth = 4
                        if (i*self.dims[0]*self.dims[1] + j, action.endStateIndex) == currentAction:
                            color = (0,255,0) # Current animation path
                    else:
                        color = Color('lightgray')
                        if gradient: # for colors floodfill
                            score = self.states[i*self.dims[0]*self.dims[1] + j].value / POINTS['R']
                            if self.states[i*self.dims[0]*self.dims[1] + j].tag != 'P':
                                color = (255 - (255 * score), 255 * score, 255 - (255 * score))
                        bwidth = 1
                    # Last layer fade
                    # Draw all arrows, even from punishment?
                    if self.states[i*self.dims[0]*self.dims[1] + j].tag != 'P' or drawFromPunishment:
                        if self.states[i*self.dims[0]*self.dims[1] + j].tag != 'R' or drawFromReward:
                            draw_arrow(screen, pygame.Vector2(j*(W+M)+M + W//2,i*(W+M)+M+ W//2),
                                pygame.Vector2((action.endStateIndex%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(i+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                            draw_arrow(screen, pygame.Vector2(j*(W+M)+M + W//2,-1*(W+M)+M+ W//2),
                                pygame.Vector2((action.endStateIndex%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(-1+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)

        def hoverMouseActions():
            """
            Hover your mouse over a state to see what actions you can take
            """
            mposx, mposy = pygame.mouse.get_pos()
            stateindx, stateindy = mposx // (W+M), mposy // (W+M)
            stateindx, stateindy = min((self.dims[0])*(self.dims[1])-1,stateindx), min(self.dims[2]-1,stateindy)
            actions = self.actions[stateindy*self.dims[0]*self.dims[1] + stateindx]
            for action in actions:
                color = Color('blue')
                bwidth = 4
                
                if stateindy != self.dims[2]-1:
                    draw_arrow(screen, pygame.Vector2(stateindx*(M+W)+M + W//2,stateindy*(M+W)+M + W//2),
                            pygame.Vector2((action.endStateIndex%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,action.endStateIndex//(self.dims[0]*self.dims[1])*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                else: # top bottom
                    draw_arrow(screen, pygame.Vector2(stateindx*(W+M)+M + W//2,stateindy*(W+M)+M+ W//2),
                            pygame.Vector2((action.endStateIndex%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(stateindy+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                    draw_arrow(screen, pygame.Vector2(stateindx*(W+M)+M + W//2,-1*(W+M)+M+ W//2),
                            pygame.Vector2((action.endStateIndex%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(-1+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)


        # -------- Main Program Loop -----------
        while not done:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        mouseHover = not mouseHover
                    elif event.key == pygame.K_SPACE:
                        autoAnimation = not autoAnimation
                    elif event.key == pygame.K_LEFT:
                        autoAnimation = False
                        c -= 1
                    elif event.key == pygame.K_RIGHT:
                        autoAnimation = False
                        c += 1
            
            screen.fill(BLACK)
            
            # Draws all states
            grid = self.grids[c%self.dims[2]]
            drawStatesAnimation(grid)
            drawStates()
            
            # Mark arrows if there is a path
            if path != None:
                if c != len(path)-1:
                    drawActions([(path[i],path[i+1]) for i in range(len(path)-1)],gradient=False,currentAction=(path[c],path[c+1]))
                else:
                    drawActions([(path[i],path[i+1]) for i in range(len(path)-1)],gradient=False)

            else:
                drawActions()


            # Mouse to show actions
            if mouseHover: hoverMouseActions()

            # bar marks down arrows
            pygame.draw.rect(screen, (0,0,0), [0, (W+M)*self.dims[2]+M,(W+M)*self.dims[0]*self.dims[1]+M,W])

            # Blit to screen
            pygame.display.update()

            clock.tick(1)
            if autoAnimation: c += 1
            if path != None:
                c = (c+len(path))%len(path)
        
        # Close the window and quit.
        pygame.quit()

    def summary(self):
        print(f"==== SUMMARY ====")
        print(f"Dimensions: {self.dims}")
        print(f"Number of states: {self.nrStates}")
        print(f"Number of actions: {self.nrActions}")
        print(f"States:")
        for ind,state in enumerate(self.states):
            print(f"State: {ind}, tag: {state.tag}, value: {state.value:.2f}, actions: {[c.endStateIndex for c in self.actions[ind]]}")


def main():
    model = Model()
    model.useFolder("grids")
    model.__setup__(moveToItself=False)
    
    model.assignGridValues()
    model.greedyPolicy()

    model.summary()

    #model.drawAnimation(path=model.path)
    #model.drawModel()
    model.drawFull(path=model.path)


if __name__ == "__main__":
    main()