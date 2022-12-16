import numpy as np
import pygame
from pygame.locals import *
from random import uniform, choice
import csv

class Environment:

    def __init__(self,folder=None):
        Environment.POINTS = {
            "P": -10,
            "R": 100,
            "S": 0,
            ".": 0,
            "X": 0
        }
        Environment.TERMINAL = [
            'R', 'P'
        ]
        self.grids = None
        self.dims = None
        self.folder = folder
        
        self.nrStates = None
        self.states = None
        self.nrActions = None
        self.actions = None
        self.startingState = None

        self.Q = None
        self.totalGenerations = 0

    # =============== #
    #      STATE      #
    # =============== #

    class State:
        def __init__(self):
            self.value = None
            self.tag = None
            self.isTerminal = None
            self.actions = {}
        
        def setTag(self,tag):
            self.tag = tag
            self.value = Environment.POINTS[tag]
            self.isTerminal = tag in Environment.TERMINAL

    # =============== #
    #      SETUP      #
    # =============== #

    def defineActions(self,actions):
        self.actions = actions
        self.nrActions = len(actions)

    def setup(self):

        def __readFiles__():

            grids = []
            dims = None
            
            with open(f"{self.folder}//0.txt","r") as f:
                dims = [int(x) for x in f.readline().split()]
                
                grids.append([[x for x in line.rstrip()] for line in f.readlines()])
                
            c = 1
            done = False
            while not done:
                try:
                    with open(f"{self.folder}//{c}.txt") as f:
                        grids.append([[x for x in line.rstrip()] for line in f.readlines()])
                    c += 1
                except:
                    done = True
            
            return dims+[c], grids
        
        def __flattenStates__():
            self.states = [[0 for _ in range(self.dims[0]*self.dims[1])] for __ in range(self.dims[2])]
            
            for grid in range(self.dims[2]):
                for row in range(self.dims[1]):
                    for col in range(self.dims[0]):
                        self.states[grid][row*self.dims[0] + col] = self.State()
                        self.states[grid][row*self.dims[0] + col].setTag(self.grids[grid][row][col])
                        if self.states[grid][row*self.dims[0] + col].tag == 'S':   # Set starting state
                            self.startingState = (grid,row*self.dims[0] + col)

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
                    for move in self.actions:
                        # Move is within grid
                        if 0 <= col+move[0] < self.dims[0] and 0 <= row+move[1] < self.dims[1]:
                            pos[move] = normalState[row+move[1]][col+move[0]]
                    
                    # Copy found moves for each dimension (gridsize is static)
                    for grid in range(self.dims[2]):
                        # Skip if terminal:
                        if self.states[grid][(row*self.dims[0]) + col].tag in self.TERMINAL:
                            self.states[grid][(row*self.dims[0]) + col].actions = {}
               
                        # Check if cell is not marked with X:
                        elif self.states[grid][(row*self.dims[0]) + col].tag != "X":
                            actions = {key: ((grid+1)%self.dims[2], value) for key,value in pos.items()}
                            actions = {key: value for key,value in actions.items() if self.states[value[0]][value[1]].tag != 'X'}  # Filters out blocked states
                            self.states[grid][(row*self.dims[0]) + col].actions = actions 
                        else:
                            self.states[grid][(row*self.dims[0]) + col].actions = {}

        if self.folder == None:
            raise Exception("No folder found!")

        if self.actions == None:
            raise Exception("No actions defined!")

        self.dims, self.grids = __readFiles__()
        self.nrStates = self.dims[0]*self.dims[1]*self.dims[2]
        self.Q = [[0 for _ in range(self.nrActions)] for _ in range(self.nrStates)]
        __flattenStates__()
        __defineActions__()
            
    # =============== #
    #      DRAW       #
    # =============== #

    def printEnvironment(self):
        # First print the grid
        print("==========")
        print("MAP:")
        print("==========")
        for time in self.states:
            for state in time:
                print(state.tag,end="")
            print()
        print("==========")
        print("ACTIONS:")
        print("==========")
        for time,row in enumerate(self.states):
            for ind, state in enumerate(row):
                print((time,ind),self.getActions((time,ind)))
    
    def draw(self,drawAnimation=True,drawModel=True,train=None,args=[]):
        """
        Draws the state-transition diagram
        Could be gigantic, beware!
        If a path is generated (for example greedyPolicy), it will draw
        the arrows of transition in gold
        """
        ##### PYGAME INIT ######
        BLACK = (0, 0, 0)
        W,M = 30,10
        pygame.init()
        if drawModel and drawAnimation:
            size = ((W+M)*self.dims[0]*self.dims[1]+M, (W+M)*self.dims[2]+M + (W+M)*self.dims[1]+M+W)
        elif drawModel:
            size = ((W+M)*self.dims[0]*self.dims[1]+M, (W+M)*self.dims[2]+M)
        elif drawAnimation:
            size = ((W+M)*self.dims[0]+M, (W+M)*self.dims[1]+M)
        else:
            raise Exception("Nothing to draw")

        screen = pygame.display.set_mode(size)
        done = False
        clock = pygame.time.Clock()
        clocktick = 1
        ########################

        # Functionalities:
        mouseHover = False           # To show actions of state you are currently hovering over
        autoAnimation = True         # Play animation
        c = 0                        # Current time step
        moves = self.getPath()       # Add empty move, to show last state

        heatmapAnimation = [[0 for _ in range(self.dims[0])] for _ in range(self.dims[1])]
        heatmap = [[0 for _ in range(self.dims[0]*self.dims[1])] for _ in range(self.dims[2])]

        def movesToPath(m):
            # Convert moves into path
            if m != None:
                marked = [self.startingState]
                for i in m:
                    marked.append(self.doAction(marked[-1],i))
            else:
                marked = []
            return marked
        
        def drawStatesAnimation(grid,state):
            """
            Draws all the states of the current layer
            """
            # Get maximum heatmap value:
            maxHeat = min(max([max(row) for row in heatmapAnimation]),255)

            if drawModel:
                exHeight = (W+M)*self.dims[2]+M + W
                exWidth = ((W+M)*self.dims[0]*self.dims[1]+M)/2 - ((W+M)*self.dims[0]+M)/2
            else:
                exHeight, exWidth = 0,0

            for i in range(self.dims[1]):
                for j in range(self.dims[0]):
                    if (grid[i][j] == "S" and path == None) or (state[1]//self.dims[0],state[1]%self.dims[0]) == (i,j):
                        pygame.draw.rect(screen, (255,255,0), [j*(W+M)+M+exWidth,i*(W+M)+M+exHeight,W,W])
                        if autoAnimation: # If not paused, or it will count the state multiple times 
                            if heatmapAnimation[i][j] < 255:
                                heatmapAnimation[i][j] += 1  
                    elif grid[i][j] == "P":
                        pygame.draw.rect(screen, (255,0,0), [j*(W+M)+M+exWidth,i*(W+M)+M+exHeight,W,W]) 
                    elif grid[i][j] == "R":
                        pygame.draw.rect(screen, (0,255,0), [j*(W+M)+M+exWidth,i*(W+M)+M+exHeight,W,W]) 
                    elif grid[i][j] == "X":
                        pass
                    else:
                        if heatmapAnimation[i][j] == 0:
                            color = (255,255,255)
                        else:
                            color = (255-int((heatmapAnimation[i][j]/maxHeat)*255),255-int((heatmapAnimation[i][j]/maxHeat)*255),255)
                        pygame.draw.rect(screen, color, [j*(W+M)+M+exWidth,i*(W+M)+M+exHeight,W,W]) 
                               
        def drawStates():
            """
            Draws all states in a self.dims[0]*self.dims[1] by self.dims[2] grid
            """
            maxHeat = min(max([max(row) for row in heatmap]),255)

            for i in range(self.dims[2]):
                for j in range(self.dims[0]*self.dims[1]):
                    # Draw state
                    if self.states[i][j].tag == "S":
                        pygame.draw.rect(screen, (255,255,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    elif self.states[i][j].tag == "P":
                        pygame.draw.rect(screen, (255,0,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    elif self.states[i][j].tag == "R":
                        pygame.draw.rect(screen, (0,255,0), [j*(W+M)+M,i*(W+M)+M,W,W]) 
                    elif self.states[i][j].tag == "X":
                        pass
                    else:
                        if heatmap[i][j] == 0:
                            color = (255,255,255)
                        else:
                            color = (255-int((heatmap[i][j]/maxHeat)*255),255-int((heatmap[i][j]/maxHeat)*255),255)
                        pygame.draw.rect(screen,color, [j*(W+M)+M,i*(W+M)+M,W,W])
            
            if autoAnimation: # If not paused, or it will count the state multiple times 
                heatmap[path[c][0]][path[c][1]] = min(heatmap[path[c][0]][path[c][1]]+self.dims[2], 255)

        def drawActions(markedArrows=[],currentAction=[]):
            """
            Draws for each state all actions. The last layer goes over to the first layer (gridwise)
            :param markedArrows: list actions in consecutive order (default = [])
            """
            for i in range(self.dims[2]-1):
                for j in range(self.dims[0]*self.dims[1]):        
                    # Draw actions
                    for action in self.states[i][j].actions.values():
                        # Path arrows color
                        if (i,j) in currentAction and action in currentAction:
                            color = (0,255,0)
                            bwidth = 4
                        elif (i,j) in markedArrows and action in markedArrows:
                            color = (255,255,0)
                            bwidth = 4
                        else:
                            color = Color('lightgray')
                            bwidth = 1
                        
                        # Draw all arrows, even from punishment?
                        draw_arrow(screen, pygame.Vector2(j*(W+M)+M + W//2,i*(W+M)+M+ W//2),
                            pygame.Vector2((action[1]*(W+M)+M+ W//2,action[0]*(W+M)+M+ W//2))
                            ,color,head_width=16,head_height=8,body_width=bwidth)
            
            
            # First/Last layer actions
            i = self.dims[2]-1
            for j in range(self.dims[0]*self.dims[1]):        
                # Draw actions
                for action in self.states[i][j].actions.values():
                    if (i,j) in currentAction and action in currentAction:
                        color = (0,255,0)
                        bwidth = 4
                    elif (i,j) in markedArrows and action in markedArrows:
                        color = (255,255,0)
                        bwidth = 4
                    else:
                        color = Color('lightgray')
                        bwidth = 1
                    # Last layer fade
                    # Draw all arrows, even from punishment?
                    draw_arrow(screen, pygame.Vector2(j*(W+M)+M + W//2,i*(W+M)+M+ W//2),
                        pygame.Vector2((action[1]*(W+M)+M+ W//2,(i+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                    draw_arrow(screen, pygame.Vector2(j*(W+M)+M + W//2,-1*(W+M)+M+ W//2),
                        pygame.Vector2((action[1]*(W+M)+M+ W//2,(-1+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)

        def hoverMouseActions():
            """
            Hover your mouse over a state to see what actions you can take
            """
            mposx, mposy = pygame.mouse.get_pos()
            stateindx, stateindy = mposx // (W+M), mposy // (W+M)
            stateindx, stateindy = min((self.dims[0])*(self.dims[1])-1,stateindx), min(self.dims[2]-1,stateindy)
            actions = self.states[stateindy][stateindx].actions.values()
            for action in actions:
                color = Color('blue')
                bwidth = 4
                
                if stateindy != self.dims[2]-1:
                    draw_arrow(screen, pygame.Vector2(stateindx*(M+W)+M + W//2,stateindy*(M+W)+M + W//2),
                            pygame.Vector2((action[1]*(W+M)+M+ W//2,action[0]*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                else: # top bottom
                    draw_arrow(screen, pygame.Vector2(stateindx*(W+M)+M + W//2,stateindy*(W+M)+M+ W//2),
                            pygame.Vector2((action[1]*(W+M)+M+ W//2,(stateindy+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
                    draw_arrow(screen, pygame.Vector2(stateindx*(W+M)+M + W//2,-1*(W+M)+M+ W//2),
                            pygame.Vector2((action[1]*(W+M)+M+ W//2,(-1+1)*(W+M)+M+ W//2)),color,head_width=16,head_height=8,body_width=bwidth)
        
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

        path = movesToPath(moves)
        moves.append((0,0))
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
                        c = (c-1+len(path))%len(path)
                    elif event.key == pygame.K_RIGHT:
                        autoAnimation = False
                        c = (c+1+len(path))%len(path)
                    elif event.key == pygame.K_UP:
                        clocktick += 5
                    elif event.key == pygame.K_DOWN:
                        clocktick = max(clocktick-5,1)

            
            screen.fill(BLACK)
            
            # Draws all states
            grid = self.grids[c%self.dims[2]]
            if drawAnimation:
                drawStatesAnimation(grid,path[c])
            if drawModel:
                drawStates()
                # Mark arrows if there is a path
                if c+1 < len(path):
                    drawActions(path,(path[c],path[c+1]))
                else:
                    drawActions(path)

            # Mouse to show actions
            if mouseHover and drawModel: hoverMouseActions()

            # bar marks down arrows
            pygame.draw.rect(screen, (0,0,0), [0, (W+M)*self.dims[2]+M,(W+M)*self.dims[0]*self.dims[1]+M,W])

            # Blit to screen
            pygame.display.update()

            clock.tick(clocktick)

            # Animation time + 1
            if autoAnimation: c += 1

            # reset animation if path exists
            if path != None:
                c = (c+len(path))%len(path)
                if c == 0 and train != None:
                    train(*args)
                    moves = self.getPath()
                    path = movesToPath(moves)
                    moves.append((0,0))
                        

        
        # Close the window and quit.
        pygame.quit()

    # =============== #
    #     POLICY      #
    # =============== #

    def updatePolicy(self,method,
                    learningRate=0.9,
                    discountFactor=0.9,
                    epsilon=0.1,
                    iterations=1
                    ):
        """
        Master function calling policy functions
        """
        if method == "Bellman":
            self.Bellman()
        elif method == "SARSA":
            self.SARSA(alpha=learningRate,
                            gamma=discountFactor,
                            epsilon=epsilon,
                            iterations=iterations)
        elif method == "QLearning":
            self.QLearning(alpha=learningRate,
                            gamma=discountFactor,
                            epsilon=epsilon,
                            iterations=iterations)

    def getQ(self,s,a):
        """
        s is a tuple (t,s)
        """
        # If you are in a terminal state, return 0
        if self.states[s[0]][s[1]].isTerminal:
            return 0
        
        # If move not possible:
        if not a in self.states[s[0]][s[1]].actions.keys():
            return 0
        
        # New state
        time, action = self.states[s[0]][s[1]].actions[a]
        if self.states[time][action].isTerminal:
            # If your move is to a terminal state, return value of the terminal state instead
            return self.states[time][action].value

        return self.Q[s[0]*self.dims[0]*self.dims[1] + s[1]][self.actions.index(a)]

    def setQ(self,s,a,value):
        self.Q[s[0]*self.dims[0]*self.dims[1] + s[1]][self.actions.index(a)] = value

    def optimalQ(self, s):
        """
        Get the optimal action from a current state
        """
        return max((self.getQ(s,a),a) for a in self.getActions(s))[1]

    def Bellman(self):

        alpha = 0.9 # learning rate
        gamma = 0.9 # discount factor
        iterations = 100
        
        for i in range(iterations):

            for time in range(self.dims[2]):
                for col in range(self.dims[0]*self.dims[1]):

                    for action in self.getActions((time,col)):
                        
                        qa = self.getQ((time,col),action)
                        reward = Environment.POINTS[self.states[time][col].tag]

                        timeprime, colprime = self.doAction((time,col),action)
                        maxExpectedFutureReward = max(self.getQ((timeprime,colprime),actionprime) for actionprime in self.getActions((timeprime,colprime)))

                        # Set new Q(S,A)
                        newqsa = qa + alpha*(reward + gamma*maxExpectedFutureReward - qa)
                        self.setQ((time, col), action, newqsa)

                    
        
        print(f"Performed Bellman equation, {iterations} iterations.")
        self.totalGenerations += iterations
        self.__exportToCSV__()

    def SARSA(self, alpha, gamma, epsilon, iterations):

        def pickAction(s):
            
            # Check if terminal:
            if self.states[s[0]][s[1]].isTerminal:
                return None
            
            randomNumber = uniform(0,1)
            
            if randomNumber <= epsilon: # random action
                a = choice(list(self.getActions(s)))
            else:                       # greedy action
                a = max((self.getQ(s,a),a) for a in self.getActions(s))[1]

            return a
        
        for iteration in range(iterations):
            # initialize S
            state = self.startingState
            # Choose A from S using e-greedy
            action = pickAction(state)
            # While not terminal
            while not self.states[state[0]][state[1]].isTerminal: 
                # Take action A, observe R, S'(=A)
                
                stateprime = self.doAction(state,action)
                reward = Environment.POINTS[self.states[stateprime[0]][stateprime[1]].tag]

                
                # Choose A' from S' using e-greedy
                actionprime = pickAction(stateprime)

                # Q(S,A) = Q(S,A) + a[R + yQ(S',A') - Q(S,A)] # NOT IN TERMINAL STATE:
                maxExpectedFutureReward = self.getQ(stateprime,actionprime)

                newqsa = self.getQ(state,action) + alpha * (reward + gamma*maxExpectedFutureReward - self.getQ(state,action))
                self.setQ(state,action,newqsa)

                # Update states and actions
                state = stateprime
                action = actionprime
        
        print(f"Performed SARSA equation, {iterations} iterations.")
        self.totalGenerations += iterations
        self.__exportToCSV__()
    
    def QLearning(self, alpha, gamma, epsilon, iterations):
        
        def pickAction(s):
            
            # Check if terminal:
            if self.states[s[0]][s[1]].isTerminal:
                return None
            
            randomNumber = uniform(0,1)
            
            if randomNumber <= epsilon: # random action
                a = choice(list(self.getActions(s)))
            else:                       # greedy action
                a = max((self.getQ(s,a),a) for a in self.getActions(s))[1]

            return a
        
        for iteration in range(iterations):
            # initialize S
            state = self.startingState
            
            # While not terminal
            while not self.states[state[0]][state[1]].isTerminal: 
                
                # Choose A from S using e-greedy
                action = pickAction(state)

                # Take action A, observe R, S'(=A)
                stateprime = self.doAction(state,action)
                reward = Environment.POINTS[self.states[stateprime[0]][stateprime[1]].tag]

                # Q(S,A) = Q(S,A) + a[R + ymaxQ(S',a) - Q(S,A)] # NOT IN TERMINAL STATE:
                if list(self.getActions(stateprime)):
                    maxExpectedFutureReward = max(self.getQ(stateprime,actionprime) for actionprime in self.getActions(stateprime))
                else:
                    maxExpectedFutureReward = 0

                newqsa = self.getQ(state,action) + alpha * (reward + gamma*maxExpectedFutureReward - self.getQ(state,action))
                self.setQ(state,action,newqsa)

                # Update state
                state = stateprime
        
        print(f"Performed QLearning equation, {iterations} iterations.")
        self.totalGenerations += iterations
        self.__exportToCSV__()

    # =============== #
    #     CONTROL     #
    # =============== #   

    def getActions(self,s):
        """
        get actions for a specific time and a specific state
        param t: 0 <= t < self.dims[2]
        param s: 0 <= s < self.dims[0]*self.dims[1]
        return: actions subset of self.actions
        """
        return self.states[s[0]][s[1]].actions.keys()

    def doAction(self,s,a):
        """
        returns the next state
        param s: current state
        param a: action
        """
        return self.states[s[0]][s[1]].actions[a]

    def getPath(self):
        curTime, curPos = self.startingState
        path = []
        states = [(curTime, curPos)]
        while not self.states[curTime][curPos].isTerminal \
            and len(set(states)) == len(states):
            move = self.optimalQ((curTime,curPos))
            path.append(move)
            curTime, curPos = self.doAction((curTime,curPos),move)
            states.append((curTime,curPos))

        #print(f"Moves found: {path}")
        return path

    # =============== #
    #     UTILITY     #
    # =============== #    

    def __exportToCSV__(self):
        # open the file in the write mode
        with open('policy.csv', 'w+') as f:

            # create the csv writer
            writer = csv.writer(f)

            # write a row to the csv file
            headers = ['State', 'Tag', 'Value'] + [f"Q({i})" for i in self.actions]
            writer.writerow(headers)

            # Write every row:
            for time in range(self.dims[2]):
                for col,state in enumerate(self.states[time]):
                    data = [(time,col), state.tag, state.value]
                    for item in self.actions:
                        data.append(self.getQ((time,col),item))
                    writer.writerow(data)
            
        print(f"Saved policy of {self.totalGenerations} generations to policy.csv")

    def summary(self,exportCSV=False,elaborate=False):
        print(f"==== SUMMARY ====")
        print(f"Dimensions: {self.dims}")
        print(f"Number of states: {self.nrStates}")
        print(f"Number of actions: {sum([len(self.getActions((i,j))) for i in range(self.dims[2]) for j in range(self.dims[0]*self.dims[1])])}")
        print(f"=================")
        if elaborate:
            print(f"States:")
            for ind, grids in enumerate(self.states):
                for jnd, state in enumerate(grids):
                    print(f"State: {(ind,jnd)}, tag: {state.tag}, value: {state.value:.2f}, actions: {state.actions}")

        if exportCSV:
            self.__exportToCSV__()

def main():
    # Setup the environment
    env = Environment("predator")
    env.defineActions([(0,1),(1,0),(-1,0),(0,-1),(0,0),(1,1),(1,-1),(-1,1),(-1,-1)])
    env.setup()
    env.updatePolicy("SARSA",
                        learningRate=0.9,
                        discountFactor=0.9,
                        epsilon=0.1,
                        iterations=1
                        )
    env.summary(exportCSV=True)

    env.draw(drawAnimation=True,
             drawModel=True,
             train=env.SARSA,
             args = (0.9,0.9,0.1,1)
            )


if __name__ == "__main__":
    main()