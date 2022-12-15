import numpy as np
import pygame
from pygame.locals import *
from random import uniform, choice
import csv

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
    "P": -100,
    "R": 100,
    "S": 0,
    ".": 0,
    "X": 0
}
TERMINAL = [
     'R', 'P'
]
        
class State:
    def __init__(self):
        self.value = None
        self.tag = None
        self.isTerminal = None

        self.actions = None
    
    def setTag(self,tag):
        self.tag = tag
        self.value = POINTS[tag]
        if tag in TERMINAL:
            self.isTerminal = True
        else:
            self.isTerminal = False

class Model:
    
    def __init__(self):
        self.grids = None
        self.dims = None
        self.folder = None
        
        self.nrStates = None
        self.states = None
        self.startingState = None
        
        self.moves = [(0,1),(0,-1),(1,0),(-1,0)]
        self.path = None
    
    def moveToItself(self,bool: bool = False):
        if bool:
            self.moves.append((0,0))
        else:
            try:
                self.moves.remove((0,0))
            except:
                pass
            
    def __setup__(self):

        def __flattenStates__():
            self.states = [0]*self.nrStates
            c = 0
            
            for grid in range(self.dims[2]):
                for row in range(self.dims[1]):
                    for col in range(self.dims[0]):
                        self.states[c] = State()
                        self.states[c].setTag(self.grids[grid][row][col])
                        if self.states[c].tag == 'S':   # Set starting state
                            self.startingState = c
                        c += 1
                
        def __defineActions__():
            normalState = []

            # Make 'identity' grid
            for i in range(self.dims[1]):
                normalState.append([])
                for j in range(self.dims[0]):
                    normalState[i].append(i*self.dims[0] + j)

            for row in range(self.dims[1]):
                for col in range(self.dims[0]):
                    
                    # Check for moves for each grid cell
                    pos = {}  
                    for move in self.moves:
                        # Move is within grid
                        if 0 <= col+move[0] < self.dims[0] and 0 <= row+move[1] < self.dims[1]:
                            pos[move] = normalState[row+move[1]][col+move[0]]

                    
                    # Copy found moves for each dimension (gridsize is static)
                    for grid in range(self.dims[2]):
                        # Skip if terminal:
                        if self.states[normalState[row][col]+(grid*(self.dims[0]*self.dims[1]))].tag in TERMINAL:
                            self.states[normalState[row][col]+(grid*(self.dims[0]*self.dims[1]))].actions = {}
               
                        # Check if cell is not marked with X:
                        elif self.states[normalState[row][col]+(grid*(self.dims[0]*self.dims[1]))].tag != "X":
                            actions = {key: value+(((grid+1)%self.dims[2])*(self.dims[0]*self.dims[1])) for key,value in pos.items()}
                            actions = {key: value for key,value in actions.items() if self.states[value].tag != 'X'}  # Filters out blocked states
                            self.states[normalState[row][col]+(grid*(self.dims[0]*self.dims[1]))].actions = actions 
                        else:
                            self.states[normalState[row][col]+(grid*(self.dims[0]*self.dims[1]))].actions = {}
    
        self.dims, self.grids = self.readFiles(self.folder)
        self.nrStates = self.dims[0]*self.dims[1]*self.dims[2]
        
        __flattenStates__()
        __defineActions__()

    def assignGridValues(self, mode,
        iterations = 10,
        epsilon = 0.1,
        learningRate = 0.9,
        discountFactor = 0.9):
        """
        Dynamically assigns values to states
        """

        def floodFill():
            change = True
            learningRate = 0.9

            iterations = 0
            
            while change:
                change = False

                for state in range(self.nrStates):
                    for action in self.states[state].actions.values():
                        newValue = self.states[action].value * learningRate
                        if self.states[state].value < newValue and self.states[state].tag != "P":
                            self.states[state].value = newValue
                            change = True
                
                iterations += 1

            print(f"Took {iterations} iterations.")

        def SARSA(iterations, epsilon, learningRate, discountFactor):
            
            def Q(st,at):
                print(st,at)
                if self.states[st].isTerminal:
                    return 0
                return self.states[at].value - self.states[st].value
            
            def pickAction(s):
                randomNumber = uniform(0,1)
                print(randomNumber,s,end=" ")
                if randomNumber <= epsilon: # random action
                    if list(self.states[s].actions.values()):
                        a = choice(list(self.states[s].actions.values()))
                    else:
                        a = None
                else:                       # greedy action
                    if list(self.states[s].actions.values()):
                        a = max((self.states[a].value,a) for a in self.states[s].actions.values())[1]
                    else:
                        a = None
                return a
            
            for iteration in range(iterations):
                # initialize S
                state = self.startingState
                # Choose A from S using e-greedy
                action = pickAction(state)
                # While not terminal
                while not self.states[state].isTerminal: 
                    # Take action A, observe R, S'(=A)
                    reward = Q(state,action)
                    stateprime = action
                    # Choose A' from S' using e-greedy
                    actionprime = pickAction(stateprime)
                    # Q(S,A) = Q(S,A) + a[R + yQ(S',A') - Q(S,A)] # NOT IN TERMINAL STATE:
                    newvalue = self.states[state].value + learningRate*(reward + discountFactor*Q(stateprime,actionprime) - self.states[state].value)
                    self.states[state].value = newvalue
                    # Update states and actions
                    state = stateprime
                    action = actionprime
                    
        if mode == "floodFill":
            floodFill()
        elif mode == "SARSA":
            SARSA(iterations, epsilon, learningRate, discountFactor)

    def makePath(self):
        self.path = [self.startingState]
        print(self.path)
        # While last state is no terminal
        while self.states[self.path[-1]].tag not in TERMINAL \
            and len(set(self.path)) == len(self.path): # duplicate
            print(self.path)

            next = max((self.states[a].value,a) for a in self.states[self.path[-1]].actions.values())[1]
            self.path.append(next)
        
        print(f"Optimal path: {self.path}")

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
                    for action in self.states[i*self.dims[0]*self.dims[1] + j].actions.values():
                        # Path arrows color
                        if (i*self.dims[0]*self.dims[1] + j, action) in markedArrows:
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
                                    pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,action//(self.dims[0]*self.dims[1])*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
            
            
            # First/Last layer actions
            i = self.dims[2]-1
            for j in range(self.dims[0]*self.dims[1]):        
                # Draw actions
                for action in self.states[i*self.dims[0]*self.dims[1] + j].actions.values():
                    if (i*self.dims[0]*self.dims[1] + j, action) in markedArrows:
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
                                pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(i+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                            draw_arrow(screen, pygame.Vector2(j*(W+M)+M + W//2,-1*(W+M)+M+ W//2),
                                pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(-1+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)

        def hoverMouseActions():
            """
            Hover your mouse over a state to see what actions you can take
            """
            mposx, mposy = pygame.mouse.get_pos()
            stateindx, stateindy = mposx // (W+M), mposy // (W+M)
            stateindx, stateindy = min((self.dims[0])*(self.dims[1])-1,stateindx), min(self.dims[2]-1,stateindy)
            actions = self.states[stateindy*self.dims[0]*self.dims[1] + stateindx].actions.values()
            for action in actions:
                color = Color('blue')
                bwidth = 4
                
                if stateindy != self.dims[2]-1:
                    draw_arrow(screen, pygame.Vector2(stateindx*(M+W)+M + W//2,stateindy*(M+W)+M + W//2),
                            pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,action//(self.dims[0]*self.dims[1])*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                else: # top bottom
                    draw_arrow(screen, pygame.Vector2(stateindx*(W+M)+M + W//2,stateindy*(W+M)+M+ W//2),
                            pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(stateindy+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                    draw_arrow(screen, pygame.Vector2(stateindx*(W+M)+M + W//2,-1*(W+M)+M+ W//2),
                            pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(-1+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)


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
                    for action in self.states[i*self.dims[0]*self.dims[1] + j].actions.values():
                        # Path arrows color
                        if (i*self.dims[0]*self.dims[1] + j, action) in markedArrows:
                            color = (255,255,0)
                            bwidth = 4
                            if (i*self.dims[0]*self.dims[1] + j, action) == currentAction:
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
                                    pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,action//(self.dims[0]*self.dims[1])*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
            
            
            # First/Last layer actions
            i = self.dims[2]-1
            for j in range(self.dims[0]*self.dims[1]):        
                # Draw actions
                for action in self.states[i*self.dims[0]*self.dims[1] + j].actions.values():
                    if (i*self.dims[0]*self.dims[1] + j, action) in markedArrows:
                        color = (255,255,0)
                        bwidth = 4
                        if (i*self.dims[0]*self.dims[1] + j, action) == currentAction:
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
                                pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(i+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                            draw_arrow(screen, pygame.Vector2(j*(W+M)+M + W//2,-1*(W+M)+M+ W//2),
                                pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(-1+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)

        def hoverMouseActions():
            """
            Hover your mouse over a state to see what actions you can take
            """
            mposx, mposy = pygame.mouse.get_pos()
            stateindx, stateindy = mposx // (W+M), mposy // (W+M)
            stateindx, stateindy = min((self.dims[0])*(self.dims[1])-1,stateindx), min(self.dims[2]-1,stateindy)
            actions = self.states[stateindy*self.dims[0]*self.dims[1] + stateindx].actions.values()
            for action in actions:
                color = Color('blue')
                bwidth = 4
                
                if stateindy != self.dims[2]-1:
                    draw_arrow(screen, pygame.Vector2(stateindx*(M+W)+M + W//2,stateindy*(M+W)+M + W//2),
                            pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,action//(self.dims[0]*self.dims[1])*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                else: # top bottom
                    draw_arrow(screen, pygame.Vector2(stateindx*(W+M)+M + W//2,stateindy*(W+M)+M+ W//2),
                            pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(stateindy+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                    draw_arrow(screen, pygame.Vector2(stateindx*(W+M)+M + W//2,-1*(W+M)+M+ W//2),
                            pygame.Vector2((action%(self.dims[0]*self.dims[1])*(W+M)+M+ W//2,(-1+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)


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

    def summary(self,exportCSV=False):
        
        def exportToCSV():
            # open the file in the write mode
            with open('policy.csv', 'w+') as f:

                # create the csv writer
                writer = csv.writer(f)

                # write a row to the csv file
                headers = ['State', 'Tag', 'Value'] + self.moves
                writer.writerow(headers)

                # Write every row:
                for ind,state in enumerate(self.states):
                    data = [ind, state.tag, state.value]
                    for item in self.moves:
                        data.append(state.actions.get(item,"-"))
                    writer.writerow(data)

        
        
        print(f"==== SUMMARY ====")
        print(f"Dimensions: {self.dims}")
        print(f"Number of states: {self.nrStates}")
        print(f"Number of actions: {sum(len(self.states[i].actions) for i in range(self.nrStates))}")
        print(f"States:")
        for ind,state in enumerate(self.states):
            print(f"State: {ind}, tag: {state.tag}, value: {state.value:.2f}, actions: {state.actions}")

        if exportCSV:
            exportToCSV()

def main():
    model = Model()
    model.useFolder("gridsclear")
    model.moveToItself(False)

    model.__setup__()
    
    model.assignGridValues(
        mode='SARSA',
        iterations=100, 
        learningRate=0.9,
        discountFactor=0.9,
        epsilon=0.1)
    
    model.makePath()
    model.summary(exportCSV=True)

    #model.drawAnimation(path=model.path)
    #model.drawModel()
    model.drawFull(path=model.path)


if __name__ == "__main__":
    main()