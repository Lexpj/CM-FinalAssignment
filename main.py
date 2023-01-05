from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import pygame
from pygame.locals import *
from random import uniform, choice, randint, shuffle
import csv
import matplotlib.pyplot as plt
import sys
from time import perf_counter, strftime, gmtime

class Environment:

    def __init__(self,folder: str = None, backToStartOnP: bool = False) -> None:
        """
        Parameters
        ----------
        `folder`:`str`, the environment folder that is used
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
        `method`:`str`, the method the rat is using for updating the policy, which is one of `['SARSA','QLearning']`
        `policyargs`:`tuple(float,float,float)`, the floats representing the learning rate, discount factor and epsilon respectively
        `cumulativeRewards`:`list`, list of all rewards through all episodes
        `randomQ`:`bool`, whether the Q(s,a) values are randomly initialized or not
        `Q`:`list(list())`, the 2D array of Q(s,a) values
        `totalGenerations`,`int`: total amount of episodes this rat has run
        `senseTime`:`bool`, whether there is a sense of time in the environment
        If this is the case, the `startingState` is always the unique state `S` found in
        the environment at a specific time. This comes down to a static `startingState = (time, pos)` 
        If this is not the case, the `startingState`'s position is the same, but time may differ.
        This comes down to `startingState = (t, pos)` for random `t` in `[0,dims[2]]`
        """
        def __init__(self,env) -> None:
            self.env = env
            self.method = None
            self.policyargs = None
            self.cumulativeRewards = None
            self.randomQ = None
            self.Q = None
            self.totalGenerations = None
            self.senseTime = None

        # =============== #
        #     UPDATE      #
        # =============== # 

        def importPolicy(self,policy: str) -> None:
            """
            Imports a CSV file of a policy\n
            Parameters
            --------
            `policy`:`str`, filename of the policy.
            """
            with open(policy, newline='') as csvfile:

                rows = csv.reader(csvfile, delimiter=',')

                rows = list(rows)

                indexes = {}
                for action in rows[0][3:]:
                    try:
                        a = tuple(int(x) for x in action[3:-2].split(','))
                        indexes[action] = self.env.actions.index(a)
                    except Exception as e:
                        print(e)

                for row in rows[1:]:
                    for i in range(3,len(row)):
                        try:
                            s = tuple(int(x) for x in row[0][1:-1].split(','))
                            a = self.env.actions[indexes[rows[0][i]]]
                            self.__setQ__(s, a, float(row[i]))
                        except:
                            pass

                print("Success!")

        def setUpdatingPolicy(self,method: str, args: tuple, randomQ: bool, senseTime: bool = True) -> None:
            """
            Sets the policy method used to train\n
            Parameters
            --------
            `method`:`str`, method in `["SARSA","QLearning"]`
            `args`:`tuple`, tuple in the form of `(alpha, gamma, epsilon)`
            `randomQ`:`bool`, whether all states start on 0 or random
            `senseTime`:`bool`, whether the rat has a sense of time (default = True)
            """
            self.method = method  
            self.policyargs = args   
            self.cumulativeRewards = [] #set at least 1 rat
            self.totalGenerations = 0
            self.randomQ = randomQ
            self.senseTime = senseTime

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
            - The set `method` must be in `['SARSA','QLearning']`\n
            Parameters:
            --------
            `episodes`:`int`: the amount of iterations to train for
            """
            if self.method == None:
                raise Exception("setUpdatingPolicy(method,args,randomQ) must be run before updatePolicy()")

            learningRate, discountFactor, epsilon = self.policyargs

            if self.method == "SARSA":
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

        def __getQ__(self,s: tuple, a: tuple) -> float:
            """
            Gets the q value of state s:(t,s) and action a.\n
            Parameters
            --------
            `s`:`tuple`, the state, consisting of (time,state)
            `a`:`tuple`, the action, consisting of (dw, dh)\n
            Returns
            --------
            `float`: Q value
            """
            # If you are in a terminal state, return 0
            if self.env.states[s[0]][s[1]].isTerminal:
                return 0
        
            return self.Q[s[0]*self.env.dims[0]*self.env.dims[1] + s[1]][self.env.actions.index(a)]

        def __setQ__(self,s,a,value) -> None:
            """
            Sets the Q `value` at a state `s` and action `a`. \n
            Parameters
            ---------
            `s`:`tuple`, the state, consisting of (time,state)
            `a`:`tuple`, the action, consisting of (dw, dh)\n
            `value`:`float`, the new value at Q(s,a)
            """
            self.Q[s[0]*self.env.dims[0]*self.env.dims[1] + s[1]][self.env.actions.index(a)] = value

        def __trueMax__(self, s) -> tuple:
            """
            Gets the action for some state that has the maximum Q(s,a) value. 
            If multiple actions in some state have the same maximum Q(s,a) value, a random action from the maximum actions is picked.\n
            Parameters
            --------
            `s`:`tuple`, the state, consisting of (time,state)\n
            Returns
            --------
            `tuple`: the action with the maximum Q(s,a) value
            """
            maxValue = max(self.__getQ__(s,a) for a in self.__getActions__(s))
            filteredLst = [a for a in self.__getActions__(s) if self.__getQ__(s,a) == maxValue]
            return choice(filteredLst)

        def __SARSA__(self, alpha, gamma, epsilon, iterations, output = False) -> None:
            """
            Updating method SARSA. 
            """
            def pickAction(s):
                
                # Check if terminal:
                if self.env.states[s[0]][s[1]].isTerminal:
                    return None
                
                randomNumber = uniform(0,1)
                
                if randomNumber <= epsilon: # random action
                    a = choice(list(self.__getActions__(s)))
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
                    stateprime = self.__doAction__(state,action)
                    reward = Environment.POINTS[self.env.states[stateprime[0]][stateprime[1]].tag]
                    
                    # Choose A' from S' using e-greedy
                    actionprime = pickAction(stateprime)

                    # Q(S,A) = Q(S,A) + a[R + yQ(S',A') - Q(S,A)] # NOT IN TERMINAL STATE:
                    maxExpectedFutureReward = self.__getQ__(stateprime,actionprime)

                    newqsa = self.__getQ__(state,action) + alpha * (reward + gamma*maxExpectedFutureReward - self.__getQ__(state,action))
                    
                    # If sense of time, just set the Q(s,a) for current t
                    if self.senseTime:
                        self.__setQ__(state,action,newqsa)
                    else: # else it should set Q(s,a) for every t 
                        for _ in range(self.env.dims[2]):
                            self.__setQ__((_,state[1]),action,newqsa)

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
        
        def __QLearning__(self, alpha, gamma, epsilon, iterations, output = False) -> None:
            """
            Updating method Q-Learning
            """
            def pickAction(s):
                
                # Check if terminal:
                if self.env.states[s[0]][s[1]].isTerminal:
                    return None
                
                randomNumber = uniform(0,1)
                
                if randomNumber <= epsilon: # random action
                    a = choice(list(self.__getActions__(s)))
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
                    stateprime = self.__doAction__(state,action)
                    reward = Environment.POINTS[self.env.states[stateprime[0]][stateprime[1]].tag]

                    # Q(S,A) = Q(S,A) + a[R + ymaxQ(S',a) - Q(S,A)] # NOT IN TERMINAL STATE:
                    if list(self.__getActions__(stateprime)):
                        maxExpectedFutureReward = max(self.__getQ__(stateprime,actionprime) for actionprime in self.__getActions__(stateprime))
                    else:
                        maxExpectedFutureReward = 0

                    newqsa = self.__getQ__(state,action) + alpha * (reward + gamma*maxExpectedFutureReward - self.__getQ__(state,action))
                    
                    # If sense of time, just set the Q(s,a) for current t
                    if self.senseTime:
                        self.__setQ__(state,action,newqsa)
                    else: # else it should set Q(s,a) for every t 
                        for _ in range(self.env.dims[2]):
                            self.__setQ__((_,state[1]),action,newqsa)
                    
                    cumReward += reward #* (gamma**self.totalGenerations)

                    # Update state
                    state = stateprime
                
                self.totalGenerations += 1
                self.cumulativeRewards.append(cumReward)
            
            if output:
                print()
                print(f"Performed QLearning equation, {iterations} iterations.")

        def __exportToCSV__(self,output=False) -> None:
            """
            Exports the Q(s,a) values of this rat to a CSV file
            """
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
        #     CONTROL     #
        # =============== #   

        def __getActions__(self,s) -> list:
            """
            Get actions for a specific time and a specific state\n
            Parameters
            --------
            `s`:`tuple`, the state, consisting of (time,state)\n
            Returns
            --------
            subset of self.actions
            """
            return self.env.states[s[0]][s[1]].actions.keys()

        def __doAction__(self,s,a) -> tuple:
            """
            Returns the next state if action `a` is done in state `s`\n
            Parameters
            --------
            `s`:`tuple`, the state, consisting of (time,state)
            `a`:`tuple`, the action, consisting of (dw, dh)\n
            Returns
            --------
            `tuple`: the next state 
            """
            return self.env.states[s[0]][s[1]].actions[a]
        
        def __getPath__(self) -> list:
            """
            Returns a path in the form of [state, state, state ...]
            The path is created with an egreedy policy.
            """
            marked = [self.env.__getStartingState__()]

            while not self.env.states[marked[-1][0]][marked[-1][1]].isTerminal:
                
                # pick e-greedy action 
                randomNumber = uniform(0,1)
                if randomNumber <= self.policyargs[2]: # random action
                    a = choice(list(self.__getActions__(marked[-1])))
                else:                       # greedy action
                    a = self.__trueMax__(marked[-1])

                newState = self.__doAction__(marked[-1],a)
                marked.append(newState)
            return marked

        # =============== #
        #     UTILITY     #
        # =============== #

        def summary(self) -> None:
            """
            Prints information of the rat
            """
            print(f"==== SUMMARY ====")
            print(f"Method: {self.method}")
            print(f"Learning rate: {self.policyargs[0]}")
            print(f"Discount factor: {self.policyargs[1]}")
            print(f"Epsilon: {self.policyargs[2]}")
            print(f"Total episodes: {self.totalGenerations}")
            print(f"=================")
            print(f"Random Q: {self.randomQ}")
            print(f"Sense of time: {self.senseTime}")
            print(f"=================")

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
        self.rats = []
        self.nrRats = 0
        self.startingTime = self.__getStartingState__()[0]
            
    # =============== #
    #      DRAW       #
    # =============== #

    def plotPerformance(self, selections = []) -> None:
        """
        Takes in a list of selections of the form (custom, label) of rats, yielded by __customSelection__, and compares
        their ACR's per episode in a plot

        Parameters:
        --------
        `selections`:`list`, list of tuples of the form (custom selection, label)\n
        Example:
        --------
        The following selections could draw 2 selections, the rats that used SARSA as update policy
        and the rats that used QLearning as update policy.\n
        `selections = [({"method":"SARSA"},"SARSA"), ({"method":"QLearning"},"QLearning")]`
        """

        maxlen = 0
        # Get and plot all ACR's        
        for item in selections:
            ACR = self.getACR(item[0])
            maxlen = max(maxlen, len(ACR)) # for xlim plot
            plt.plot(list(range(0,len(ACR))),ACR,label=item[1])

        # Setup up plot
        plt.ylim([min(Environment.POINTS.values()), max(Environment.POINTS.values())])
        plt.xlim(0,maxlen)
        plt.legend()
        plt.xlabel("Episodes")
        plt.ylabel("Sum of rewards")
        plt.show()

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
    
    def draw(self, rat: Rat, drawAnimation: bool = True, drawModel: bool = True) -> None:
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
        path = rat.__getPath__()    # Add empty move, to show last state
        c = 0                        # Loop through steps

        heatmapAnimation = [[0 for _ in range(self.dims[0])] for _ in range(self.dims[1])]
        heatmap = [[0 for _ in range(self.dims[0]*self.dims[1])] for _ in range(self.dims[2])]

        def pathToMoves(p):
            # Convert moves into path
            marked = []
            if p != None:
                for i in range(len(p)-1):
                    marked.append((-(p[i+1][0]%self.dims[0]-p[i][0]%self.dims[0]),-(p[i+1][1]//self.dims[0]-p[i][1]//self.dims[0])))
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
                    if path[0] == (i,j):
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

        moves = pathToMoves(path)
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
                if c == 0:
                    path = rat.__getPath__()
                    moves = pathToMoves(path)
                    moves.append((0,0))
                        

        
        # Close the window and quit.
        pygame.quit()
    
    def draw3D(self,rat=None) -> None:
        """
        Draws a matplotlib model in 3D\n
        Parameters:
        --------
        `path`:`bool`, whether the model should draw the path obtained from `__getPath__()`
        """

        if rat != None:
            cells = rat.__getPath__()
            moves = [[a for a in rat.__getActions__(cells[i]) if rat.__doAction__(cells[i],a) == cells[i+1]][0] for i in range(len(cells)-1)]

        def getColors(i):
            lst = []
            for ind, row in enumerate(self.grids[i]):
                colors = []
                for jnd, col in enumerate(row):
                    if rat == None and col == 'S':
                        colors.append(np.array([1,1,0]))
                    elif rat != None and (i,ind*self.dims[0]+jnd) == cells[0]:
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
        for time in range(self.dims[2]):
            for state in range(self.dims[1]*self.dims[0]):
                for action in self.states[time][state].actions:
                    if time == self.dims[2]-1: # end => start
                        ax.quiver(state%self.dims[0] + 0.5,time+1,state//self.dims[0] + 0.5,
                            action[0],0.5,action[1],color=[0.9,0.9,0.9],alpha=0.2)
                        ax.quiver(state%self.dims[0] + 0.5,0.5,state//self.dims[0] + 0.5,
                            action[0],0.5,action[1],color=[0.9,0.9,0.9],alpha=0.2)
                    else:
                        ax.quiver(state%self.dims[0] + 0.5,time+1,state//self.dims[0] + 0.5,
                            action[0],1,action[1],color=[0.9,0.9,0.9],alpha=0.2)
        
        # Draw this over the other arrows
        if rat != None:
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

    # =============== #
    #      RATS       #
    # =============== #

    def __customSelection__(self,args: dict) -> list:
        """
        Gets the selection of rats that meet a specific requirement. Possible custom tags:
        {
            "method": str,
            "alpha": float,
            "gamma": float,
            "epsilon": float,
            "episodes": int,
            "randomQ": bool,
            "senseTime": bool
        }
        Selection are a list of indeces
        """
        rats = list(range(self.nrRats))

        # Method
        method = args.get("method",None)
        if method != None:
            rats = [i for i in rats if self.rats[i].method == method]
        
        # Alpha
        alpha = args.get("alpha",None)
        if alpha != None:
            rats = [i for i in rats if self.rats[i].policyargs[0] == alpha]
        
        # Gamma
        gamma = args.get("gamma",None)
        if gamma != None:
            rats = [i for i in rats if self.rats[i].policyargs[1] == gamma]
        
        # Epsilon
        epsilon = args.get("epsilon",None)
        if epsilon != None:
            rats = [i for i in rats if self.rats[i].policyargs[2] == epsilon]
        
        # Episodes
        episodes = args.get("episodes",None)
        if episodes != None:
            rats = [i for i in rats if self.rats[i].totalGenerations == episodes]
        
        # randomQ
        randomQ = args.get("randomQ",None)
        if randomQ != None:
            rats = [i for i in rats if self.rats[i].randomQ == randomQ]
        
        # senseTime
        senseTime = args.get("senseTime",None)
        if senseTime != None:
            rats = [i for i in rats if self.rats[i].senseTime == senseTime]

        print(f"Custom selection yielded {len(rats)} rat(s)")
        return rats

    def getACR(self, custom = None, all = True) -> list:
        """
        Gets the average cumulative reward for a selection of rats, if not all
        """
        if custom != None:
            ratInd = self.__customSelection__(custom)
        elif all:
            ratInd = list(range(self.nrRats))

        maxEpisodes = max(self.rats[i].totalGenerations for i in ratInd)
        lst = [[] for i in range(maxEpisodes)]
        for ind in ratInd:
            for i in range(self.rats[ind].totalGenerations):
                lst[i].append(self.rats[ind].cumulativeRewards[i])
        return [sum(x)/len(x) for x in lst]

    def killRats(self, custom = None, all = True) -> None:
        """
        Kills a selection of rats, if not all
        """
        if self.nrRats == None or self.nrRats == 0:
            return

        if custom != None:
            ratInd = self.__customSelection__(custom)
        elif all:
            ratInd = []

        keepalive = list(set(range(self.nrRats)) - set(ratInd))

        self.rats = [self.rats[rat] for rat in keepalive]
        self.nrRats = len(self.rats)
    
    def generateRats(self,rats: int, method: str, args: tuple, randomQ: bool, senseTime: bool = True) -> None:
        """
        Generates rats\n
        Parameters
        --------\n
        `rats`:`int`: the amount of rats generated
        `method`:`str`, the method the rat is using for updating the policy, which is one of `['Bellman','SARSA','QLearning']`
        `args`:`tuple(float,float,float)`, the floats representing the learning rate, discount factor and epsilon respectively
        `randomQ`:`bool`, whether the Q(s,a) values are randomly initialized or not        
        `senseTime`:`bool`, whether there is a sense of time in the environment
        """
        for rat in range(rats):
            self.rats.append(self.Rat(self))
            self.rats[-1].setUpdatingPolicy(method, args, randomQ, senseTime)
        self.nrRats += rats

    def trainRats(self, episodes, custom = None, all = True) -> None:
        """
        Trains rats by updating their Q(s,a) values\n
        Parameters
        --------\n
        `episodes`: `int`, the amount of episodes the rats train for
        `custom`: `list`, a custom selection of rats (as indexes) you want to train (default = None)
        `all`: `bool`, whether it should train all rats. Custom selections have priority
        """
        if custom != None:
            ratInd = self.__customSelection__(custom)
        elif all:
            ratInd = list(range(self.nrRats))

        ETA = 0
        recalculate = episodes//10

        for ind,rat in enumerate(ratInd):
            # Print status

            if ind%recalculate == 0: t1 = perf_counter()
            
            sys.stdout.write("\r [" + "="*int(((ind+1)/len(ratInd)) * 20) + "."*(20-(int(((ind+1)/len(ratInd)) * 20))) +f"] RATS={ind+1}/{len(ratInd)}, ETA: {ETA}")
            sys.stdout.flush()
            
            self.rats[rat].updatePolicy(episodes)
            
            if ind%recalculate == 0: t2 = perf_counter()

            ETA = (t2-t1) * (len(ratInd) - ind)
            ETA = strftime("%M:%S", gmtime(ETA))
        print()

    def averageQRats(self, custom = None, all = True) -> Rat:
        """
        Returns the a rat containing the average Q(s,a) values of a selection of rats\n
        Parameters
        ----------
        `custom`: `list`, a custom selection of rats (as indexes) you want to average (default = None)
        `all`: `bool`, whether it should average all rats. Custom selections have priority\n
        Return
        --------
        A rat of type `Rat`.
        IT IS ASSUMED THE Q(S,A) TABLE OF ALL RATS HAS THE SAME SHAPE
        IF ANY VARIABLES VARY OVER ALL RATS (E.G. METHOD, POLICY ARGS, ETC.), IT TAKES THE FIRST RAT
        """    
        if custom != None:
            ratInd = self.__customSelection__(custom)
        elif all:
            ratInd = list(range(self.nrRats))

        
        avgRat = self.Rat(self)
        nrAvgRats = len(ratInd)

        if nrAvgRats == 0:
            return avgRat
        
        avgRat.setUpdatingPolicy(self.rats[ratInd[0]].method,self.rats[ratInd[0]].policyargs,randomQ = False,senseTime=self.rats[ratInd[0]].senseTime)

        for rat in ratInd:
            for time in range(self.dims[2]):
                for states in range(self.dims[1]*self.dims[0]):
                    for action in self.actions:
                        avgRat.__setQ__((time,states),action, \
                            avgRat.__getQ__((time,states),action) + self.rats[rat].__getQ__((time,states),action)/nrAvgRats)
        return avgRat

    # =============== #
    #     CONTROL     #
    # =============== #

    def __getStartingState__(self) -> tuple:
        """
        Get a random starting state
        """
        self.startingTime = randint(0,self.dims[2]-1)
        return (self.startingTime,self.startingState[1])

    # =============== #
    #     UTILITY     #
    # =============== #    

    def summary(self,elaborate=False) -> None:

        def getRatConfigs():
            """
            Extracts rat configs as keys and amount of rats having that config as values
            """
            dic = {}
            for rat in self.rats:
                info = (rat.method, rat.policyargs, rat.totalGenerations, rat.randomQ, rat.senseTime)
                dic[info] = dic.get(info, 0) + 1
            return dic

        if self.dims == None:
            raise Exception("Please setup() the environment first!")

        print(f"==== SUMMARY ====")
        print(f"Dimensions: {self.dims}")
        print(f"Number of states: {self.nrStates}")
        print(f"Number of actions: {sum([len(self.states[i][j].actions) for i in range(self.dims[2]) for j in range(self.dims[0]*self.dims[1])])}")
        print(f"=================")
        print(f"Using folder: {self.folder}")
        print(f"Possible moves: {self.nrActions}")
        print(f"Moves: {self.actions}")
        print(f"Back to start on P state: {self.backToStartOnP}")
        print(f"Starting state: {self.startingState}")
        print(f"=================")
        print(f"Number of rats: {self.nrRats}")
        print(f"Unique rat configurations: {len(getRatConfigs().keys())}")
        ratconfigs = getRatConfigs()
        for config in ratconfigs.keys():
            print(f" | {ratconfigs[config]} rats with:")
            print(f" | - Method: {config[0]}")
            print(f" | - Learning rate: {config[1][0]}")
            print(f" | - Discount factor: {config[1][1]}")
            print(f" | - Epsilon: {config[1][2]}")
            print(f" | - Episodes: {config[2]}")
            print(f" | - randomQ: {config[3]}")
            print(f" | - senseTime: {config[4]}")
        print(f"=================")
        if elaborate:
            print(f"States:")
            for ind, grids in enumerate(self.states):
                for jnd, state in enumerate(grids):
                    print(f"State: {(ind,jnd)}, tag: {state.tag}, value: {state.value:.2f}, actions: {state.actions}")



def SuttonAndBarto():
    """
    Example 6.6 from Reinforcement Learning by Sutton and Barto.
    Done in 7 lines
    """
    env = Environment("cliffworld",backToStartOnP=True)
    env.defineActions([(0,1),(1,0),(-1,0),(0,-1)])
    env.setup()

    env.generateRats(100, method="SARSA", args=(0.1,0.9,0.1), randomQ=True, senseTime=True)
    env.generateRats(100, method="QLearning", args=(0.1,0.9,0.1), randomQ=True, senseTime=True)
    env.trainRats(episodes=500,all=True)

    env.plotPerformance([
        ({"method":"SARSA"},"SARSA"),
        ({"method":"QLearning"},"QLearning")
    ])

def Environment1():
    """
    Environment 1, seen in the result section of the paper 
    """
    env = Environment("env1",backToStartOnP=False)
    env.defineActions([(0,1),(1,0),(-1,0),(0,-1),(0,0)]) # <- Note (0,0). Rats may stand still
    env.setup()
    env.draw3D()
    rats = 100
    episodes = 500

    # SARSA rats with sense of time
    env.generateRats(rats, method="SARSA", args=(0.1,0.9,0.1), randomQ=True, senseTime=True)
    # SARSA rats without sense of time
    env.generateRats(rats, method="SARSA", args=(0.1,0.9,0.1), randomQ=True, senseTime=False)
    # QLearning rats with sense of time
    env.generateRats(rats, method="QLearning", args=(0.1,0.9,0.1), randomQ=True, senseTime=True)
    # QLearning rats without sense of time
    env.generateRats(rats, method="QLearning", args=(0.1,0.9,0.1), randomQ=True, senseTime=False)
    # Train all rats
    env.trainRats(episodes=episodes,all=True)

    # Plot performances
    env.plotPerformance([
        ({"method": "SARSA", "senseTime": True}, "SARSA with time"),
        ({"method": "SARSA", "senseTime": False}, "SARSA without time"),
        ({"method": "QLearning", "senseTime": True}, "QLearning with time"),
        ({"method": "QLearning", "senseTime": False}, "QLearning without time")
    ])


    # To draw the animations, like stated in the results, remove the return:
    return

    # Get average rats
    avgRatSARSATIME = env.averageQRats({"method": "SARSA", "senseTime": True})
    avgRatQLearningTIME = env.averageQRats({"method": "QLearning", "senseTime": True})
    avgRatSARSANOTIME = env.averageQRats({"method": "SARSA", "senseTime": False})
    avgRatQLearningNOTIME = env.averageQRats({"method": "QLearning", "senseTime": False})

    # Average rat with sense of time using SARSA
    env.draw(avgRatSARSATIME)
    # Average rat with sense of time using QLearning
    env.draw(avgRatQLearningTIME)
    # Average rat without sense of time using SARSA
    env.draw(avgRatSARSANOTIME)
    # Average rat without sense of time using QLearning
    env.draw(avgRatQLearningNOTIME)

def Environment2():
    """
    Environment 2, seen in the result section of the paper 
    """
    env = Environment("env2",backToStartOnP=False)
    env.defineActions([(0,1),(1,0),(-1,0),(0,-1)]) # <- Note no (0,0). Rats may not stand still
    env.setup()
    env.draw3D()
    rats = 100
    episodes = 500

    # SARSA rats with sense of time
    env.generateRats(rats, method="SARSA", args=(0.1,0.9,0.1), randomQ=True, senseTime=True)
    # SARSA rats without sense of time
    env.generateRats(rats, method="SARSA", args=(0.1,0.9,0.1), randomQ=True, senseTime=False)
    # QLearning rats with sense of time
    env.generateRats(rats, method="QLearning", args=(0.1,0.9,0.1), randomQ=True, senseTime=True)
    # QLearning rats without sense of time
    env.generateRats(rats, method="QLearning", args=(0.1,0.9,0.1), randomQ=True, senseTime=False)
    # Train all rats
    env.trainRats(episodes=episodes,all=True)

    # Plot performances
    env.plotPerformance([
        ({"method": "SARSA", "senseTime": True}, "SARSA with time"),
        ({"method": "SARSA", "senseTime": False}, "SARSA without time"),
        ({"method": "QLearning", "senseTime": True}, "QLearning with time"),
        ({"method": "QLearning", "senseTime": False}, "QLearning without time")
    ])


    # To draw the animations, like stated in the results, remove the return:
    return

    # Get average rats
    avgRatSARSATIME = env.averageQRats({"method": "SARSA", "senseTime": True})
    avgRatQLearningTIME = env.averageQRats({"method": "QLearning", "senseTime": True})
    avgRatSARSANOTIME = env.averageQRats({"method": "SARSA", "senseTime": False})
    avgRatQLearningNOTIME = env.averageQRats({"method": "QLearning", "senseTime": False})

    # Average rat with sense of time using SARSA
    env.draw(avgRatSARSATIME)
    # Average rat with sense of time using QLearning
    env.draw(avgRatQLearningTIME)
    # Average rat without sense of time using SARSA
    env.draw(avgRatSARSANOTIME)
    # Average rat without sense of time using QLearning
    env.draw(avgRatQLearningNOTIME)

def Environment3():
    env = Environment("env3",backToStartOnP=False)
    env.defineActions([(0,1),(1,0),(-1,0),(0,-1),(0,0)]) # <- Note (0,0). Rats may stand still
    env.setup()
    env.draw3D()
    rats = 100
    episodes = 500

    # SARSA rats with sense of time
    env.generateRats(rats, method="SARSA", args=(0.1,0.9,0.1), randomQ=True, senseTime=True)
    # SARSA rats without sense of time
    env.generateRats(rats, method="SARSA", args=(0.1,0.9,0.1), randomQ=True, senseTime=False)
    # QLearning rats with sense of time
    env.generateRats(rats, method="QLearning", args=(0.1,0.9,0.1), randomQ=True, senseTime=True)
    # QLearning rats without sense of time
    env.generateRats(rats, method="QLearning", args=(0.1,0.9,0.1), randomQ=True, senseTime=False)
    # Train all rats
    env.trainRats(episodes=episodes,all=True)

    # Plot performances
    env.plotPerformance([
        ({"method": "SARSA", "senseTime": True}, "SARSA with time"),
        ({"method": "SARSA", "senseTime": False}, "SARSA without time"),
        ({"method": "QLearning", "senseTime": True}, "QLearning with time"),
        ({"method": "QLearning", "senseTime": False}, "QLearning without time")
    ])


    # To draw the animations, like stated in the results, remove the return:
    return

    # Get average rats
    avgRatSARSATIME = env.averageQRats({"method": "SARSA", "senseTime": True})
    avgRatQLearningTIME = env.averageQRats({"method": "QLearning", "senseTime": True})
    avgRatSARSANOTIME = env.averageQRats({"method": "SARSA", "senseTime": False})
    avgRatQLearningNOTIME = env.averageQRats({"method": "QLearning", "senseTime": False})

    # Average rat with sense of time using SARSA
    env.draw(avgRatSARSATIME)
    # Average rat with sense of time using QLearning
    env.draw(avgRatQLearningTIME)
    # Average rat without sense of time using SARSA
    env.draw(avgRatSARSANOTIME)
    # Average rat without sense of time using QLearning
    env.draw(avgRatQLearningNOTIME)

def EnvironmentDiscussion():
    env = Environment("calc",backToStartOnP=False)
    env.defineActions([(0,1),(1,0),(-1,0),(0,-1),(0,0)]) # <- Note (0,0). Rats may stand still
    env.setup()
    env.draw3D()
    rats = 100
    episodes = 500

    # SARSA rats with sense of time
    env.generateRats(rats, method="SARSA", args=(0.1,0.9,0.1), randomQ=True, senseTime=True)
    # SARSA rats without sense of time
    env.generateRats(rats, method="SARSA", args=(0.1,0.9,0.1), randomQ=True, senseTime=False)
    # QLearning rats with sense of time
    env.generateRats(rats, method="QLearning", args=(0.1,0.9,0.1), randomQ=True, senseTime=True)
    # QLearning rats without sense of time
    env.generateRats(rats, method="QLearning", args=(0.1,0.9,0.1), randomQ=True, senseTime=False)
    # Train all rats
    env.trainRats(episodes=episodes,all=True)

    # Plot performances
    env.plotPerformance([
        ({"method": "SARSA", "senseTime": True}, "SARSA with time"),
        ({"method": "SARSA", "senseTime": False}, "SARSA without time"),
        ({"method": "QLearning", "senseTime": True}, "QLearning with time"),
        ({"method": "QLearning", "senseTime": False}, "QLearning without time")
    ])


    # To draw the animations, like stated in the results, remove the return:
    

    # Get average rats
    avgRatSARSATIME = env.averageQRats({"method": "SARSA", "senseTime": True})
    avgRatQLearningTIME = env.averageQRats({"method": "QLearning", "senseTime": True})
    avgRatSARSANOTIME = env.averageQRats({"method": "SARSA", "senseTime": False})
    avgRatQLearningNOTIME = env.averageQRats({"method": "QLearning", "senseTime": False})

    # Average rat with sense of time using SARSA
    env.draw(avgRatSARSATIME)
    # Average rat with sense of time using QLearning
    env.draw(avgRatQLearningTIME)
    # Average rat without sense of time using SARSA
    env.draw(avgRatSARSANOTIME)
    # Average rat without sense of time using QLearning
    env.draw(avgRatQLearningNOTIME)

def compareSenses(folder):
    """
    So instead of knowing what time you are in, or being able to know that, you get
    2 scenarios:

    1. the rat starts in a random time step but knows what time it is.
    2. the rat starts in a random time step but does now know what time it is, 
        nor knows how many time steps there are. This is the same as filling
        every Q(s,a) value for every T at the same time, since, you do not know what 
        time it is. So you can actually move just like normal, but, since you adjust
        all time steps, no time step will be different from each other, thus not
        having access to memory of a specific time step AND a specific state, but just
        the specific state.
    """

    env = Environment(folder,backToStartOnP=False)
    env.defineActions([(0,1),(1,0),(-1,0),(0,-1),(0,0)])
    env.setup()
    env.draw3D()
    rats = 100
    episodes = 500

    # SARSA rats with sense of time
    env.generateRats(rats, method="SARSA", args=(0.1,0.9,0.1), randomQ=True, senseTime=True)
    # SARSA rats without sense of time
    env.generateRats(rats, method="SARSA", args=(0.1,0.9,0.1), randomQ=True, senseTime=False)
    # QLearning rats with sense of time
    env.generateRats(rats, method="QLearning", args=(0.1,0.9,0.1), randomQ=True, senseTime=True)
    # QLearning rats without sense of time
    env.generateRats(rats, method="QLearning", args=(0.1,0.9,0.1), randomQ=True, senseTime=False)
    # Train all rats
    env.trainRats(episodes=episodes,all=True)

    # Plot performances
    env.plotPerformance([
        ({"method": "SARSA", "senseTime": True}, "SARSA with time"),
        ({"method": "SARSA", "senseTime": False}, "SARSA without time"),
        ({"method": "QLearning", "senseTime": True}, "QLearning with time"),
        ({"method": "QLearning", "senseTime": False}, "QLearning without time")
    ])



    avgRatSARSA = env.averageQRats({"method": "SARSA", "senseTime": False})
    avgRatSARSA.summary()
    #avgRatSARSA.__exportToCSV__()
    avgRatQLearning = env.averageQRats({"method": "QLearning", "senseTime": False})
    avgRatQLearning.summary()
    #avgRatQLearning.__exportToCSV__()

    env.draw(avgRatSARSA)
    env.draw(avgRatQLearning)

    avgRatSARSA = env.averageQRats({"method": "SARSA", "senseTime": True})
    avgRatSARSA.summary()
    #avgRatSARSA.__exportToCSV__()
    avgRatQLearning = env.averageQRats({"method": "QLearning", "senseTime": True})
    avgRatQLearning.summary()
    #avgRatQLearning.__exportToCSV__()

    env.draw(avgRatSARSA)
    env.draw(avgRatQLearning)


def main():

    # Program

    # Sutton and Barto example 6.6
    # SuttonAndBarto()

    # Environment1() 
    # Environment2()
    # Environment3()
    

    return
  

if __name__ == "__main__":
    main()

# TODO: export images
