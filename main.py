from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import pygame
from pygame.locals import *
from random import uniform, choice, randint
import csv
import matplotlib.pyplot as plt
import sys

class Environment:

    def __init__(self,folder: str = None, senseTime: bool = True,
                backToStartOnP: bool = False) -> None:
        """
        Parameters
        ----------
        `folder`:`str`, the environment folder that is used
        `senseTime`:`bool`, whether there is a sense of time in the environment
        If this is the case, the `startingState` is always the unique state `S` found in
        the environment at a specific time. This comes down to a static `startingState = (time, pos)` 
        If this is not the case, the `startingState`'s position is the same, but time may differ.
        This comes down to `startingState = (t, pos)` for random `t` in `[0,dims[2]]`
        `backToStartOnP`:`bool`, episode doesn't end on a P tagged state, but sends you back to the start.
        This comes with the prerequisite that P is not in `Environment.TERMINAL`.
        """
        Environment.POINTS = {
            "P": -100,
            "R": 100,
            "S": -1,
            ".": -1,
            "X": 0
        }
        Environment.TERMINAL = ['R'] if backToStartOnP else ['R','P']
        self.grids = None
        self.dims = None
        self.folder = folder
        
        self.nrStates = None
        self.states = None
        self.nrActions = None
        self.actions = None
        self.backToStartOnP = backToStartOnP
        self.startingState = None
        self.startingTime = None
        self.senseTime = senseTime

        self.rats = None
        self.nrRats = None

    # =============== #
    #  HELP CLASSES   #
    # =============== #

    class State:
        """
        State class. Holds a few variables:
        `value`:`int`, the value of a state defined by its `tag`
        `tag`:`str`, the tag of a state. Can be any of `['.','P','R','S','X']`
        `isTerminal`:`bool`, whether the state is an end state
        `actions`:`dic{move:state}`, all possible actions along with their next state
        """
        def __init__(self) -> None:
            self.value = None
            self.tag = None
            self.isTerminal = None
            self.actions = {}
        
        def setTag(self,tag: str) -> None:
            """
            Sets the tag of a state, along with the corresponding `value` and `isTerminal` values
            Parameters
            ----------
            `tag`:`str`: the tag of a state. Can be any of `['.','P','R','S','X']`
            """
            self.tag = tag
            self.value = Environment.POINTS[tag]
            self.isTerminal = tag in Environment.TERMINAL
    
    class Rat:
        """
        Rat class. Holds a few variables:
        `env`:`Environment`, the current environment
        `method`:`str`, the method the rat is using for updating the policy, which is one of `['Bellman','SARSA','QLearning']`
        `policyargs`:`tuple(float,float,float)`, the floats representing the learning rate, discount factor and epsilon respectively
        `cumulativeRewards`:`list`, list of all rewards through all episodes
        `randomQ`:`bool`, whether the Q(s,a) values are randomly initialized or not
        `Q`:`list(list())`, the 2D array of Q(s,a) values
        `totalGenerations`,`int`: total amount of episodes this rat has run
        """
        def __init__(self,env):
            self.env = env
            self.method = None
            self.policyargs = None
            self.cumulativeRewards = None
            self.randomQ = None
            self.Q = None
            self.totalGenerations = None
            
        def importPolicy(self,policy):
            pass

        def setUpdatingPolicy(self,method: str, args: tuple, randomQ: bool) -> None:
            """
            Sets the policy method used to train\n
            Parameters
            --------
            `method`:`str`, method in `["Bellman","SARSA","QLearning"]`
            `args`:`tuple`, tuple in the form of `(alpha, gamma, epsilon)`
            `randomQ`:`bool`, whether all states start on 0 or random
            """
            self.method = method  
            self.policyargs = args   
            self.cumulativeRewards = [] #set at least 1 rat
            self.totalGenerations = 0
            self.randomQ = randomQ

            if randomQ:
                # random initialization of the Q values
                self.Q = [[randint(0, max(Environment.POINTS.values())) for _ in range(self.env.nrActions)] for _ in range(self.env.nrStates)]
                
                # However, terminal states = 0
                for time in range(self.env.dims[2]):
                    for state in range(self.env.dims[1]*self.env.dims[0]):
                        if self.env.states[time][state].isTerminal:
                            for action in range(self.env.nrActions):
                                self.Q[time*self.env.dims[1]*self.env.dims[0] + state][action] = 0
            else:
                self.Q = [[0 for _ in range(self.env.nrActions)] for _ in range(self.env.nrStates)]

        def resetPolicy(self) -> None:
            """
            Resets the policy. A new policy has to be set for training
            using `setUpdatingPolicy(method, args, randomQ)`\n
            Prerequisites:
            --------
            - `resetPolicy()` can only be run after `setup()`
            """
            if self.env.nrStates == None:
                raise Exception("resetPolicy() can only be run after setup()")
            
            self.Q = None
            self.totalGenerations = None
            self.method = None
            self.policyargs = None
            self.cumulativeRewards = None
            self.rats = None

        def updatePolicy(self, episodes: int) -> None:
            """
            Master function calling policy functions\n
            Prerequisites:
            --------
            - `setUpdatingPolicy(method,args,randomQ)` must be run before `updatePolicy()`\n
            - The set `method` must be in `['Bellman','SARSA','QLearning']`\n
            Parameters:
            --------
            `episodes`:`int`: the amount of iterations to train for
            """
            if self.method == None:
                raise Exception("setUpdatingPolicy(method,args,randomQ) must be run before updatePolicy()")

            learningRate, discountFactor, epsilon = self.policyargs

            if self.method == "Bellman":
                self.__Bellman__()
            elif self.method == "SARSA":
                self.__SARSA__(alpha=learningRate,
                                gamma=discountFactor,
                                epsilon=epsilon,
                                iterations=episodes)
            elif self.method == "QLearning":
                self.__QLearning__(alpha=learningRate,
                                gamma=discountFactor,
                                epsilon=epsilon,
                                iterations=episodes)
            else:
                raise Exception(f"Method {self.method} not found")

        def __getQ__(self,s,a):
            """
            s is a tuple (t,s)
            """
            # If you are in a terminal state, return 0
            if self.env.states[s[0]][s[1]].isTerminal:
                return 0
        
            return self.Q[s[0]*self.env.dims[0]*self.env.dims[1] + s[1]][self.env.actions.index(a)]

        def __setQ__(self,s,a,value):
            self.Q[s[0]*self.env.dims[0]*self.env.dims[1] + s[1]][self.env.actions.index(a)] = value

        def __optimalQ__(self, s):
            """
            Get the optimal action from a current state
            """
            return max((self.__getQ__(s,a),a) for a in self.env.__getActions__(s))[1]

        def __trueMax__(self, s):
            maxValue = max(self.__getQ__(s,a) for a in self.env.__getActions__(s))
            filteredLst = [a for a in self.env.__getActions__(s) if self.__getQ__(s,a) == maxValue]
            return choice(filteredLst)

        def __Bellman__(self):

            alpha = 0.9 # learning rate
            gamma = 0.9 # discount factor
            iterations = 100
            
            for i in range(iterations):

                for time in range(self.env.dims[2]):
                    for col in range(self.env.dims[0]*self.env.dims[1]):

                        for action in self.env.__getActions__((time,col)):
                            
                            qa = self.__getQ__((time,col),action)
                            reward = Environment.POINTS[self.env.states[time][col].tag]

                            timeprime, colprime = self.env.__doAction__((time,col),action)
                            maxExpectedFutureReward = max(self.__getQ__((timeprime,colprime),actionprime) for actionprime in self.env.__getActions__((timeprime,colprime)))

                            # Set new Q(S,A)
                            newqsa = qa + alpha*(reward + gamma*maxExpectedFutureReward - qa)
                            self.__setQ__((time, col), action, newqsa)

                        
            
            print(f"Performed Bellman equation, {iterations} iterations.")
            self.totalGenerations += iterations
            self.__exportToCSV__()

        def __SARSA__(self, alpha, gamma, epsilon, iterations, output = False):

            def pickAction(s):
                
                # Check if terminal:
                if self.env.states[s[0]][s[1]].isTerminal:
                    return None
                
                randomNumber = uniform(0,1)
                
                if randomNumber <= epsilon: # random action
                    a = choice(list(self.env.__getActions__(s)))
                else:                       # greedy action
                    a = self.__trueMax__(s)
                return a
            
            for iteration in range(iterations):
                
                if output:
                    sys.stdout.write("\r [" + "="*int(((iteration+1)/iterations) * 20) + "."*(20-(int(((iteration+1)/iterations) * 20))) +f"] SARSA ITERATION={iteration+1}     ")
                    sys.stdout.flush()
                
                cumReward = 0
                # initialize S
                state = self.env.__getStartingState__()
                path = []
                # Choose A from S using e-greedy
                action = pickAction(state)
                # While not terminal
                while not self.env.states[state[0]][state[1]].isTerminal: 
                    # Take action A, observe R, S'(=A)
                    path.append(state)
                    stateprime = self.env.__doAction__(state,action)
                    reward = Environment.POINTS[self.env.states[stateprime[0]][stateprime[1]].tag]
                    
                    # Choose A' from S' using e-greedy
                    actionprime = pickAction(stateprime)

                    # Q(S,A) = Q(S,A) + a[R + yQ(S',A') - Q(S,A)] # NOT IN TERMINAL STATE:
                    maxExpectedFutureReward = self.__getQ__(stateprime,actionprime)

                    newqsa = self.__getQ__(state,action) + alpha * (reward + gamma*maxExpectedFutureReward - self.__getQ__(state,action))
                    self.__setQ__(state,action,newqsa)
                    cumReward += reward # * (gamma**self.totalGenerations)

                    # Update states and actions
                    state = stateprime
                    action = actionprime
                
                self.totalGenerations += 1

                # For plotting
                self.cumulativeRewards.append(cumReward)

            if output:
                print()
                print(f"Performed SARSA equation, {iterations} iterations.")
            self.__exportToCSV__(output)
        
        def __QLearning__(self, alpha, gamma, epsilon, iterations, output = False):
            
            def pickAction(s):
                
                # Check if terminal:
                if self.env.states[s[0]][s[1]].isTerminal:
                    return None
                
                randomNumber = uniform(0,1)
                
                if randomNumber <= epsilon: # random action
                    a = choice(list(self.env.__getActions__(s)))
                else:                       # greedy action
                    a = self.__trueMax__(s)

                return a
            
            for iteration in range(iterations):
                
                if output:
                    sys.stdout.write("\r [" + "="*int(((iteration+1)/iterations) * 20) + "."*(20-(int(((iteration+1)/iterations) * 20))) +f"] QLEARNING ITERATION={iteration+1}     ")
                    sys.stdout.flush()
                
                cumReward = 0
                # initialize S
                state = self.env.__getStartingState__()
                
                # While not terminal
                while not self.env.states[state[0]][state[1]].isTerminal: 
                    
                    # Choose A from S using e-greedy
                    action = pickAction(state)

                    # Take action A, observe R, S'(=A)
                    stateprime = self.env.__doAction__(state,action)
                    reward = Environment.POINTS[self.env.states[stateprime[0]][stateprime[1]].tag]

                    # Q(S,A) = Q(S,A) + a[R + ymaxQ(S',a) - Q(S,A)] # NOT IN TERMINAL STATE:
                    if list(self.env.__getActions__(stateprime)):
                        maxExpectedFutureReward = max(self.__getQ__(stateprime,actionprime) for actionprime in self.env.__getActions__(stateprime))
                    else:
                        maxExpectedFutureReward = 0

                    newqsa = self.__getQ__(state,action) + alpha * (reward + gamma*maxExpectedFutureReward - self.__getQ__(state,action))
                    self.__setQ__(state,action,newqsa)
                    
                    cumReward += reward #* (gamma**self.totalGenerations)

                    # Update state
                    state = stateprime
                
                self.totalGenerations += 1
                self.cumulativeRewards.append(cumReward)
            
            if output:
                print()
                print(f"Performed QLearning equation, {iterations} iterations.")
            self.__exportToCSV__(output)

        def __exportToCSV__(self,output=False):
            # open the file in the write mode
            with open(f'{self.method}.csv', 'w+') as f:

                # create the csv writer
                writer = csv.writer(f)

                # write a row to the csv file
                headers = ['State', 'Tag', 'Value'] + [f"Q({i})" for i in self.env.actions]
                writer.writerow(headers)

                # Write every row:
                for time in range(self.env.dims[2]):
                    for col,state in enumerate(self.env.states[time]):
                        data = [(time,col), state.tag, state.value]
                        for item in self.env.actions:
                            data.append(self.__getQ__((time,col),item))
                        writer.writerow(data)

            if output:    
                print(f"Saved policy of {self.totalGenerations} generations to {self.method}.csv")

    # =============== #
    #      SETUP      #
    # =============== #

    def defineActions(self,actions: list) -> None:
        """
        Defines the actions in this environment.\n
        Parameters
        ---------
        `actions`:`list[tuple(int,int)]`, a list of tuples of every action possible in the environment
        An action is defined as a move in the `x` direction and a move in the `y` direction. 
        A tuple is thus defined as `(dx,dy)`. 

        Example
        ---------
        A list of actions which allows the agent to move up, down, left and right is the following:\n
        `actions = [(0,1),(1,0),(0,-1),(-1,0)]`\n
        An addition to this list is a move to itself, which allows the agent to stand still:\n
        `action = [(0,1),(1,0),(0,-1),(-1,0),(0,0)]`\n
        Lastly, an agent that is able to do kingsmoves, is the following:\n
        `action = [(0,1),(1,0),(0,-1),(-1,0),(-1,-1),(1,-1),(-1,1),(1,1)]`
        """
        self.actions = actions
        self.nrActions = len(actions)

    def setup(self) -> None:
        """
        Setup of the environment. \n
        This reads the environment folder, sets up all states and connects them via actions.\n
        Furthermore, the policy is reset. This has to be independently defined later by `setUpdatingPolicy(method, args, randomQ)`
        Prerequisites
        --------
        - The `actions` are set using `defineActions(actions)`\n
        - A valid `folder` is set with at least one level `0.txt`.\n
        """

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
            
            if self.startingState == None:
                raise Exception("No start state S found!")

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

                        # If punishment and backToStart is True
                        elif self.states[grid][(row*self.dims[0]) + col].tag == 'P' and self.backToStartOnP:
                            self.states[grid][(row*self.dims[0]) + col].actions = {self.actions[0]:self.__getStartingState__()} # only move is back to start

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
        __flattenStates__()
        __defineActions__()
        self.killRats()
        self.startingTime = self.__getStartingState__()[0]
            
    # =============== #
    #      DRAW       #
    # =============== #

    def printEnvironment(self) -> None:
        """
        Prints a simplified version of the states, along with all actions
        """
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
                print((time,ind),self.__getActions__((time,ind)))
    
    def draw(self,drawAnimation: bool = True, drawModel: bool = True, train: bool = False) -> None:
        """
        Draws the state-transition diagram
        Could be gigantic, beware!\n
        Prerequisites
        ---------
        - The model is set up using `setup()`\n
        Parameters
        ---------
        `drawAnimation`:`bool`, whether the animation is drawn. 
        This animation shows the path the agent takes with the current policy. 
        `drawModel`:`bool`, whether the model is drawn. The model draws the transition
        diagram between states.
        `train`:`bool`, whether the policy is updated throughout the animation. Prerequisite is
        that a policy is set using `setPolicy(mode, args, randomQ)`. The policy is updated with `episodes=1`.  
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
        moves = self.__getMoves__()  # Add empty move, to show last state
        c = 0                        # Loop through steps

        heatmapAnimation = [[0 for _ in range(self.dims[0])] for _ in range(self.dims[1])]
        heatmap = [[0 for _ in range(self.dims[0]*self.dims[1])] for _ in range(self.dims[2])]

        def movesToPath(m):
            # Convert moves into path
            if m != None:
                marked = [(self.startingTime,self.startingState[1])]
                for i in m:
                    marked.append(self.__doAction__(marked[-1],i))
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
            grid = self.grids[(c+self.startingTime)%self.dims[2]]
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
                if c == 0 and train:
                    self.updatePolicy(1)
                    moves = self.__getMoves__()
                    path = movesToPath(moves)
                    moves.append((0,0))
                        

        
        # Close the window and quit.
        pygame.quit()
    
    def draw3D(self,path=False) -> None:
        """
        Draws a matplotlib model in 3D\n
        Parameters:
        --------
        `path`:`bool`, whether the model should draw the path obtained from `__getPath__()`
        """

        if path:
            cells = self.__getPath__()
            moves = [[a for a in self.__getActions__(cells[i]) if self.__doAction__(cells[i],a) == cells[i+1]][0] for i in range(len(cells)-1)]

        def getColors(i):
            lst = []
            for ind, row in enumerate(self.grids[i]):
                colors = []
                for jnd, col in enumerate(row):
                    if col == 'S' and self.senseTime:
                        colors.append(np.array([1,1,0]))
                    elif path and (i,ind*self.dims[0]+jnd) == cells[0]:
                        colors.append(np.array([1,1,0]))
                    elif col == 'X':
                        colors.append(np.array([0,0,0]))
                    elif col == 'R':
                        colors.append(np.array([0,1,0]))
                    elif col == 'P':
                        colors.append(np.array([1,0,0]))
                    else:
                        colors.append(np.array([0.8,0.8,0.8]))
                colors.append([0,0,0])
                lst.append(colors)
            
            # Extra grid row
            colors = []
            for col in row:
                colors.append(np.array([0,0,0]))
            colors.append(np.array([0,0,0]))
            
            lst.append(np.array(colors))
            return np.array(lst)
                    
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        X = np.arange(0,self.dims[0]+1,1)
        Z = np.arange(0,self.dims[1]+1,1)
        X, Z = np.meshgrid(X, Z)
        ax.set_xlabel('Width')
        ax.set_ylabel('Time')
        ax.set_zlabel('Height')
        ax.set_xticks([])
        ax.set_yticks([i for i in range(0,self.dims[2]+1)])
        ax.set_zticks([])
        ax.invert_zaxis()
        ax.set_box_aspect((1,5,1))
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Draw arrows
        for time,row in enumerate(self.states):
            for ind, state in enumerate(row):
                for action in self.__getActions__((time,ind)):
                    if time == self.dims[2]-1: # end => start
                        ax.quiver(ind%self.dims[0] + 0.5,time+1,ind//self.dims[0] + 0.5,
                            action[0],0.5,action[1],color=[0.9,0.9,0.9],alpha=0.2)
                        ax.quiver(ind%self.dims[0] + 0.5,0.5,ind//self.dims[0] + 0.5,
                            action[0],0.5,action[1],color=[0.9,0.9,0.9],alpha=0.2)
                    else:
                        ax.quiver(ind%self.dims[0] + 0.5,time+1,ind//self.dims[0] + 0.5,
                            action[0],1,action[1],color=[0.9,0.9,0.9],alpha=0.2)
        
        # Draw this over the other arrows
        if path:
            for i in range(len(moves)):
                if cells[i][0] == self.dims[2]-1: # end => start
                    ax.quiver(cells[i][1]%self.dims[0] + 0.5,cells[i][0]+1,cells[i][1]//self.dims[0] + 0.5,
                        moves[i][0],0.5,moves[i][1],color=[0.9,0.9,0.0])
                    ax.quiver(cells[i][1]%self.dims[0] + 0.5,0.5,cells[i][1]//self.dims[0] + 0.5,
                        moves[i][0],0.5,moves[i][1],color=[0.9,0.9,0.0])
                else:
                    ax.quiver(cells[i][1]%self.dims[0] + 0.5,cells[i][0]+1,cells[i][1]//self.dims[0] + 0.5,
                        moves[i][0],1,moves[i][1],color=[0.9,0.9,0.0])


        # Draw grids
        for i in range(self.dims[2]):
            C = getColors(i)
            ax.plot_surface(X, np.ones(shape=X.shape)+i, Z, facecolors=C, linewidth=0)
        
        plt.show()

    def plotPerformance(self) -> None:
        """
        Plots a graph of the performance of current policy
        """
        plt.plot(list(range(0,self.totalGenerations)),self.cumulativeRewards,label=f'{self.method}, senseTime = {self.senseTime}, backToStartOnP = {self.backToStartOnP}')
       
        plt.legend()
        plt.xlabel("Episodes")
        plt.ylabel("Sum of reward")
        plt.show()

    # =============== #
    #      RATS       #
    # =============== #

    def getAverageCumulativeReward(self):
        maxEpisodes = max(rat.totalGenerations for rat in self.rats)
        lst = [[] for i in range(maxEpisodes)]
        for rat in self.rats:
            for i in range(rat.totalGenerations):
                lst[i].append(rat.cumulativeRewards[i])
        return [sum(x)/len(x) for x in lst]

    def killRats(self):
        """
        Kills all rats
        """
        self.rats = []
        self.nrRats = 0
    
    def generateRats(self,rats: int, method: str, args: tuple, randomQ: bool):
        """
        Generates rats\n
        Parameters
        --------\n
        `rats`:`int`: the amount of rats generated
        `method`:`str`, the method the rat is using for updating the policy, which is one of `['Bellman','SARSA','QLearning']`
        `args`:`tuple(float,float,float)`, the floats representing the learning rate, discount factor and epsilon respectively
        `randomQ`:`bool`, whether the Q(s,a) values are randomly initialized or not        
        """
        for rat in range(rats):
            self.rats.append(self.Rat(self))
            self.rats[-1].setUpdatingPolicy(method, args, randomQ)
        self.nrRats += rats
        print()

    def trainRats(self, episodes, custom = None, all = True):
        """
        Trains rats by updating their Q(s,a) values\n
        Parameters
        --------\n
        `episodes`: `int`, the amount of episodes the rats train for
        `custom`: `list`, a custom selection of rats (as indexes) you want to train (default = None)
        `all`: `bool`, whether it should train all rats. Custom selections have priority
        """
        if custom != None:
            for index,item in enumerate(custom):
                # Print status
                sys.stdout.write("\r [" + "="*int(((index+1)/len(custom)) * 20) + "."*(20-(int(((index+1)/len(custom)) * 20))) +f"] RATS={index+1}/{len(custom)}     ")
                sys.stdout.flush()
                
                # Check if index in range:
                if item < len(self.rats):
                    self.rats[index].updatePolicy(episodes)
                else:
                    raise Exception(f"Can't address rat {item}, since there are only {len(self.rats)} rats.")
        elif all:
            for index, rat in enumerate(self.rats):
                # Print status
                sys.stdout.write("\r [" + "="*int(((index+1)/self.nrRats) * 20) + "."*(20-(int(((index+1)/self.nrRats) * 20))) +f"] RATS={index+1}/{self.nrRats}     ")
                sys.stdout.flush()
                rat.updatePolicy(episodes)
        print()
        
    # =============== #
    #     CONTROL     #
    # =============== #   

    def __getStartingState__(self):
        """
        Get a random starting state concerning time, if self.senseTime==False
        """
        if self.senseTime:
            self.startingTime = self.startingState[0]
            return self.startingState
        self.startingTime = randint(0,self.dims[2]-1)
        return (self.startingTime,self.startingState[1])

    def __getActions__(self,s):
        """
        get actions for a specific time and a specific state
        param t: 0 <= t < self.dims[2]
        param s: 0 <= s < self.dims[0]*self.dims[1]
        return: actions subset of self.actions
        """
        return self.states[s[0]][s[1]].actions.keys()

    def __doAction__(self,s,a):
        """
        returns the next state
        param s: current state
        param a: action
        """
        return self.states[s[0]][s[1]].actions[a]

    def __getMoves__(self):
        curTime, curPos = self.__getStartingState__()
        path = []
        states = [(curTime, curPos)]
        while not self.states[curTime][curPos].isTerminal \
            and len(set(states)) == len(states):
            move = self.__optimalQ__((curTime,curPos))
            path.append(move)
            curTime, curPos = self.__doAction__((curTime,curPos),move)
            states.append((curTime,curPos))

        #print(f"Moves found: {path}")
        return path
    
    def __getPath__(self):
        marked = [(self.startingTime,self.startingState[1])]
        for i in self.__getMoves__():
            marked.append(self.__doAction__(marked[-1],i))
        return marked

    # =============== #
    #     UTILITY     #
    # =============== #    

    def summary(self,exportCSV=False,elaborate=False):
        print(f"==== SUMMARY ====")
        print(f"Dimensions: {self.dims}")
        print(f"Number of states: {self.nrStates}")
        print(f"Number of actions: {sum([len(self.__getActions__((i,j))) for i in range(self.dims[2]) for j in range(self.dims[0]*self.dims[1])])}")
        print(f"=================")
        print(f"Using folder: {self.folder}")
        print(f"Possible moves: {self.nrActions}")
        print(f"Moves: {self.actions}")
        print(f"Back to start on P state: {self.backToStartOnP}")
        print(f"Sense of time: {self.senseTime}")
        print(f"Starting state: {self.startingState}")
        print(f"=================")
        print(f"Policy method: {self.method}")
        print(f"Learning rate alpha: {self.policyargs[0]}")
        print(f"Discount factor gamma: {self.policyargs[1]}")
        print(f"Epsilon: {self.policyargs[2]}")
        print(f"Q(s,a) initialized randomly: {self.randomQ}")
        print(f"Total generations trained: {self.totalGenerations}")
        print(f"=================")
        if elaborate:
            print(f"States:")
            for ind, grids in enumerate(self.states):
                for jnd, state in enumerate(grids):
                    print(f"State: {(ind,jnd)}, tag: {state.tag}, value: {state.value:.2f}, actions: {state.actions}")

        if exportCSV:
            self.__exportToCSV__()


def compareAlgorithms(folder,args,rats,episodes,randomQ):
    env = Environment(folder,backToStartOnP=True)
    env.defineActions([(0,1),(1,0),(-1,0),(0,-1)])
    env.setup()

    env.generateRats(rats, method="SARSA", args=args, randomQ=randomQ)
    env.trainRats(episodes=episodes,all=True)

    SARSA = env.getAverageCumulativeReward()

    #env.draw3D(path=True)

    env = Environment(folder,backToStartOnP=True)
    env.defineActions([(0,1),(1,0),(-1,0),(0,-1)])
    env.setup()

    env.generateRats(rats, method="QLearning", args=args, randomQ=randomQ)
    env.trainRats(episodes=episodes,all=True)
    QLEARNING = env.getAverageCumulativeReward()
    
    #env.draw3D(path=True)
    #print(SARSA,QLEARNING)
    
    plt.ylim([min(Environment.POINTS.values()), max(Environment.POINTS.values())])
    plt.xlim(0,len(SARSA))
    plt.plot(list(range(0,episodes)),[SARSA[i] for i in range(0,episodes)],label="SARSA")
    plt.plot(list(range(0,episodes)),[QLEARNING[i] for i in range(0,episodes)],label="QLearning")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards")
    plt.show()

def compareSenses(folder, args, episodes, randomQ):
    # With sense of time, using SARSA
    env = Environment(folder,senseTime=True,backToStartOnP=True)
    env.defineActions([(0,1),(1,0),(-1,0),(0,-1)])
    env.setup()

    env.setUpdatingPolicy("SARSA", args, randomQ)
    env.updatePolicy(episodes)
    timeSenseSARSA = env.cumulativeRewards.copy()

    # Without sense of time, using SARSA
    env = Environment(folder,senseTime=False,backToStartOnP=True)
    env.defineActions([(0,1),(1,0),(-1,0),(0,-1)])
    env.setup()

    env.setUpdatingPolicy("SARSA", args, randomQ)
    env.updatePolicy(episodes)
    noTimeSenseSARSA = env.cumulativeRewards.copy()

    # With sense of time, using SARSA
    env = Environment(folder,senseTime=True,backToStartOnP=True)
    env.defineActions([(0,1),(1,0),(-1,0),(0,-1)])
    env.setup()

    env.setUpdatingPolicy("QLearning", args, randomQ)
    env.updatePolicy(episodes)
    timeSenseQLEARNING = env.cumulativeRewards.copy()

    # Without sense of time, using SARSA
    env = Environment(folder,senseTime=False,backToStartOnP=True)
    env.defineActions([(0,1),(1,0),(-1,0),(0,-1)])
    env.setup()

    env.setUpdatingPolicy("QLearning", args, randomQ)
    env.updatePolicy(episodes)
    noTimeSenseQLEARNING = env.cumulativeRewards.copy()

    # Plot:
    plt.plot(list(range(0,episodes)),[timeSenseSARSA[i] for i in range(0,episodes)],label='SARSA, senseTime = True')
    plt.plot(list(range(0,episodes)),[noTimeSenseSARSA[i] for i in range(0,episodes)],label="SARSA, senseTime = False")
    plt.plot(list(range(0,episodes)),[timeSenseQLEARNING[i] for i in range(0,episodes)],label='QLearning, senseTime = True')
    plt.plot(list(range(0,episodes)),[noTimeSenseQLEARNING[i] for i in range(0,episodes)],label="QLearning, senseTime = False")
    
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Sum of reward")
    plt.show()



def main():

    compareAlgorithms("gridworld",(0.1,0.5,0.1),rats=200,episodes=500,randomQ=True)

    #compareSenses("gridworld", (0.1,0.9,0.1), episodes=100, randomQ=False)
    # Setup the environment

    return

    ## Setup the model
    env = Environment("lvl3",senseTime=True,backToStartOnP=False)
    env.defineActions([(0,1),(1,0),(-1,0),(0,-1)])
    env.setup()

    ## Set policy
    env.setUpdatingPolicy("SARSA",(0.1,0,0.1),randomQ=True)

    ## Update policy with x number of episodes
    env.updatePolicy(episodes=500)

    ## Draw method
    env.draw3D(path=False)

    ## Draw
    env.summary(exportCSV=True)

    env.draw(drawAnimation=True,
             drawModel=True,
             train=True)
    #env.plotPerformance()


if __name__ == "__main__":
    main()


# Notes van de meeting:
# <check> backToStart klopt dat?
# score is gemiddelde van multiple rats 
# <check> trueMax instead of max
# no knowledge of time means no time to know