"""
Instructions for the generator

A: horizontal, from side to side, starting moving to the left
B: horizontal, from side to side, starting moving to the right
C: vertical, from side to side, starting moving to the top
D: vertical, from side to side, starting moving to the bottom

The punishments move through the reward states and other punishing states,
however, it does not move through X.

"""


class Generator:

    def __init__(self):
        self.folder = None
        self.grid = None
        self.dims = None

    def useFolder(self,folder):
        self.folder = folder

    def readFiles(self):
        if self.folder != None:            
            with open(f"{self.folder}//0.txt","r") as f:
                self.dims = [int(x) for x in f.readline().split()]
                self.grid = [[x for x in line.rstrip()] for line in f.readlines()]

        else:
            print("Set a folder first!")

    def printInitialGrid(self):
        """
        Prints initial grid
        """
        if self.grid != None:
            for row in self.grid:
                print(''.join(row))
    
    def generate(self):
        """
        Expands the initial grid by translating the letters to moving punishments
        First it translates all letters to objects
        Then, it cycles those until it reaches the starting state
        Then, it translates all letters for every grid for every cycle
        Lastly, it exports all those grids to files
        """
        def move(item):
            lst = [item[0],item[1],item[2]]
            if item[0] == "A":

                if lst[1] == 0 or self.grid[lst[2]][lst[1]-1] == 'X':
                    lst[0] = "B"
                    lst[1] += 1
                else:
                    lst[1] -= 1
            
            elif lst[0] == "B":

                if lst[1] == self.dims[0]-1 or self.grid[lst[2]][lst[1]+1] == 'X':
                    lst[0] = "A"
                    lst[1] -= 1
                else:
                    lst[1] += 1
            
            elif lst[0] == "C":

                if lst[2] == 0 or self.grid[lst[2]-1][lst[1]] == 'X':
                    lst[0] = "D"
                    lst[2] += 1
                else:
                    lst[2] -= 1
            
            elif lst[0] == "D":

                if lst[2] == self.dims[1]-1 or self.grid[lst[2]+1][lst[1]] == 'X':
                    lst[0] = "C"
                    lst[2] -= 1
                else:
                    lst[2] += 1
            return lst

        def createObjects():
            """
            Create from all letters objects
            """
            objects = []

            for i in range(self.dims[1]):
                for j in range(self.dims[0]):
                    if self.grid[i][j] != '.':
                        objects.append([self.grid[i][j],j,i])
            return objects

        def cycleObjects(objects):
            """
            Cycle letters and return full cycle list
            """
            allObjects = [objects]
            newObjects = [move(obj) for obj in objects]

            while set(tuple(x) for x in allObjects[0]) != set(tuple(x) for x in newObjects):

                allObjects.append(newObjects)
                newObjects = [move(obj) for obj in newObjects]


            return allObjects

        def translateToGrids(cycles):
            allGrids = []

            for cycle in cycles:
                grid = [["." if self.grid[y][x] != "X" else "X" for x in range(self.dims[0])] for y in range(self.dims[1])]

                for item in cycle:
                    if item[0] in "ABCD": # Translate to punishment
                        grid[item[2]][item[1]] = "P"
                    else:
                        grid[item[2]][item[1]] = item[0]
                
                allGrids.append(grid)
            
            return allGrids
            
        def saveToFile(grids):

            for count in range(len(grids)):
                with open(f"{self.folder}//{count}.txt","w+") as f:
                    if count == 0:
                        f.write(str(self.dims[0]) + " " + str(self.dims[1])+"\n")
                    
                    for line in grids[count]:
                        f.write(''.join(line)+"\n")
                                        

        allLetters = createObjects()
        allCycles = cycleObjects(allLetters)
        allGrids = translateToGrids(allCycles)
        saveToFile(allGrids)
        #print(allGrids)

def main():
    gen = Generator()
    gen.useFolder("gen")
    gen.readFiles()
    gen.printInitialGrid()

    gen.generate()


if __name__ == "__main__":
    main()
