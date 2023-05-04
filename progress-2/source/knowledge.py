from itertools import combinations
from pysat.solvers import Glucose3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Read file
with open("input.txt") as f:
  maze = f.read()

# Each line of the maze is an element in the list
maze_list = maze.splitlines()

# Get the first line of the maze
first_line = maze_list[0].split(" ")

# Get the rows and columns2
rows = int(first_line[0])
columns = int(first_line[1])

# The maze after removing first line
maze = maze_list[1:]

# Find neighbors of a cell
def get_adjacents(r, c):
    result = []
    for row_add in range(-1, 2):
        new_row = r + row_add
        if new_row >= 0 and new_row <= len(maze)-1:
            for col_add in range(-1, 2):
                new_col = c + col_add
                if new_col >= 0 and new_col <= len(maze)-1:
                    if new_col == c and new_row == r:
                        continue
                    result.append((new_row,new_col))
    result.append((r, c))
    return result

def convert_coordinate_to_int(neighbors):
  result = []
  for neighbor in neighbors:
    result.append(neighbor[0] * rows + neighbor[1] + 1)
  return result

variables = [[i*rows+j+1 for j in range(rows)] for i in range(columns)]

clauses = []

# Each cell must be green or red
for i in range(rows):
  for j in range(columns):
    clause = [variables[i][j], -variables[i][j]]
    clauses.append(clause)

for i in range(rows):
    for j in range(columns):
        if maze[i][j] != " ":

            adjacents = convert_coordinate_to_int(get_adjacents(i, j))
            for c in combinations(adjacents, len(adjacents)-int(maze[i][j])+1):
                clauses.append(list(c))

            not_adjacents = [-i for i in adjacents]
            for c in combinations(not_adjacents, int(maze[i][j])+1):
                clauses.append(list(c))

g = Glucose3()

for clause in clauses:
    g.add_clause(clause)

g.solve()

# Get solution
solution = g.get_model()

def show_detail(boolean):
    if boolean:
        print("Clauses: \n", clauses, end="\n\n")
        print("Solution: ", solution)

def draw(boolean):
    if boolean:
        new_maze = []
        for s in solution:
            if s < 0: s = 0
            else: s = 1
            new_maze.append(s)

        new_maze = np.array(new_maze).reshape(rows, columns)

        fig, ax = plt.subplots()
        cmap = colors.ListedColormap(['Red','Green'])
        ax.matshow(new_maze, cmap=cmap)
        for i in range(rows):
            for j in range(columns):
                c = maze[j][i] # input maze
                ax.text(i, j, str(c), va='center', ha='center')

        plt.show()


draw(True)
show_detail(False)

