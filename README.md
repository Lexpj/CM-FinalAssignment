### CM Assignment

## Installation
First clone this repository. In this repository, you will also find a `requirements.txt` file. Install the necessary packages by

```bash
pip install -r requirements.txt
```

Open `main.py` to use the program.

## Usage
I will elaborately explain the usage in this section.

### Environment setup
There is a protocol to follow in order to set up such an environment. First of all, there has to be a folder that contains files numbered from `0.txt` to the amount of time steps there are. Each folder must at least have one time step, and thus at least contain `0.txt`. The first line of this file states the width _w_ and height _h_ of the environment and environments that follow. Difference in size of the environment is not allowed (However, you could state that states are inaccessible by giving them the tag 'X'). Then, _h_ lines follow with _w_ symbols, which are explained in the environment class. The environment as a whole must have only one starting state S in some time step. Every time step that follows, just contains the grid for that time step. An environment could look like this:

```
| env2
--| 0.txt
--| 1.txt
--| 2.txt
--| 3.txt
```
And each of the files contain:

```
0.txt:
3 3
S..
P..
..R

1.txt:
...
.P.
..R

2.txt:
...
..P
..R

3.txt:
...
.P.
..R
```

The environment also has the parameter _backToStartOnP_, which states that whenever you land on a P state, the agent gets sent back to the starting state. By default, the P state is a terminal state and _backToStartOnP_ is set to false.

Next up are actions. Actions are defined in this program as a list of tuples of the form $(\Delta w, \Delta h)$. These will get translated into $(t+1, \Delta w + (\Delta h \times w_{old}))$ in the program after `setup()`. The `setup()` function will read the environment from the specified folder and translate these into states of the right form (tags, values, etc.), each defined with actions that will define what state you end up in when you perform that action. This function will also kill any rats still present in the environment. A simple set up would look like:

```python
env = Environment("gridworld",backToStartOnP=False)
env.defineActions([(0,1),(1,0),(-1,0),(0,-1)])
env.setup()
```

### Rats
Rats are generated by stating how many rats you want to generate, what update function is used to train the rats (SARSA or Q-Learning), what the learning rate $\alpha$, the discount factor $\gamma$ and the $\epsilon$-greedy $\epsilon$ value are, whether all $Q(s,a)$ values are randomly initialized or not, and whether the rat has a sense of time or not. For example, take a look at the following algorithm:
```python
env.generateRats(100, method="SARSA", args=(0.1,0.9,0.1), randomQ=True, senseTime=True)
```
This line states that there are 100 rats generated, using the updating method SARSA, having an $\alpha = 0.1$, $\gamma = 0.9$ and $\epsilon = 0.1$, where all $Q(s,a)$ values are randomly initialized and the rat has a sense of time.

More rats can be generated and will all be contained in the environment. This allows, for example, to both have 100 rats updating on SARSA and 100 rats updating on Q-Learning. 
In order to train the rats, you specify what rats you want to train and for how many episodes. In the following line, all rats are trained for 500 episodes. 
```python
env.trainRats(episodes=500,all=True)
```
You can also make a custom selection of rats to be trained. For example, having defined 100 rats using SARSA and 100 rats using QLearning, the algorithm above is the same as the algorithm below:
```
env.trainRats(episodes=500,custom={"method":"SARSA"})
env.trainRats(episodes=500,custom={"method":"QLearning"})
```
The custom function filters out only the rats that apply to all arguments specified in the given dictionary.

There are a couple of visual options. First of all, the function `draw3D()` draws a three dimensional representation of the environment, like seen here:

![plot](./assets/cyclicgrid(1).png)


If the parameter `path = True`, it will show the path in yellow. 

Second, there is a `summary()` function, where it lists all the information about the model. This includes dimensions, number of states, number of actions, specified folder, moves, number of rats and many more. Each rat also has a `summary()` function, which states their arguments, total episodes, updating function, etc. 

Furthermore, there is a function to draw the model as well as the animation of how a rat with its currently trained policy would walk through the model. The path will also be highlighted in yellow. In order to do that, it is advised to make an 'average' rat. This rat contains the averaged out $Q(s,a)$ values of all rats in that selection. An example for an average rat from rats using SARSA and having a sense of time is seen in algorithm here. 
```python
averageRatSarsaTime = env.averageQRats({"method":"SARSA","senseTime":True})
env.draw(rat=averageRatSarsaTime)
```
Drawing is done using PyGame. An example of a is seen here:
![plot](./assets/animation.gif)

\begin{figure}
    \centering
    \includegraphics[scale=0.65]{modelanimation.png}
    \caption{Environment of figure \ref{fig:1} from three-dimensional to two-dimensional representation, with in yellow the path of the current rat in the environment. In the program, the bottom grid shows the animation of the rat showing its moves of the path it goes down.}
    \label{fig:modelanimation}
\end{figure}
Lastly, the performance of a selection of rats can be visualized in a plot. This is done by algorithm \ref{ACR}.
\begin{algorithm}[caption={Plots performance}, label={ACR}]
env.plotPerformance([
    ({"method":"SARSA"},"SARSA"),
    ({"method":"QLearning"},"QLearning")
])
\end{algorithm}
The function takes a list of tuples each of the form (selection, label). For example, algorithm \ref{ACR} takes the selection of all rats having an updating method "SARSA" and label them SARSA, and a selection of all rats having an updating method "QLearning", with the label QLearning. The function shows a plot with on the x-axis the number of episodes and on the y-axis the average cumulative reward of said selection of rats.

