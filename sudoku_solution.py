import numpy as np
from itertools import chain

digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits


def cross(A, B):
    squares = []
    for a in A:
        for b in B:
            squares.append(a + b)
    return squares


# every square in sudoku has 3 units
row_unitlist = []
col_unitlist = []
box_unitlist = []

# row unit
for r in rows:
    a = cross(r, cols)
    row_unitlist.append(a)

# column unit
for c in cols:
    a = cross(rows, c)
    col_unitlist.append(a)

# box unit
for rs in ('ABC', 'DEF', 'GHI'):
    for cs in ('123', '456', '789'):
        d = cross(rs, cs)
        box_unitlist.append(d)

# creating a unit list of all three units
unitlist = (row_unitlist + col_unitlist + box_unitlist)

# squares unit list
squares = []

for r in rows:
    for c in cols:
        squares.append(r + c)

# presenting each square in grid as key and its units as value
units = {}

for s in squares:
    for u in unitlist:
        if s in u:
            if s not in units:
                units[s] = []
            units[s].append(u)

# Peers - The peers of ‘C2’ is all squares in the related 3 units except for ‘C2’ itself.
peers = {}

for s in squares:
    unit_set = set()
    for unit in units[s]:
        for square in unit:
            if square != s:
                unit_set.add(square)
    peers[s] = unit_set


# testing all the values of squares, units, peers
def test():
    "A set of unit test"
    assert len(squares) == 81  # assert help in testing a certain value
    assert len(unitlist) == 27
    assert all(len(units[s]) == 3 for s in squares)
    assert all(len(peers[s]) == 20 for s in squares)
    print('All tests pass.')


test()


# extracting grid values
def extract_grid_values(grid):
    grid1_char = []
    for i in grid:
        for j in i:
            if str(j) in '0123456789':
                grid1_char.append(int(j))
    return grid1_char


#grid = extract_grid_values(str(grid))

# values in a dictionary using square as key and char values as value.
def grid_values(grid):
    grid1_values = {}
    for k, v in zip(squares, grid):
        grid1_values[k] = v
    return grid1_values


# constraint propagation
# if a square has only one possible then eliminate that value from the square peers
# If a unit has only one possible place for a value, then put the value there.

def eliminate(values, s, d):
    """eliminate d from values[s]; propagate when values or plaves <=2"""
    if d not in values[s]:
        return values  # already eliminated
    values[s] = values[s].replace(d, '')

    # (1) --> if a square s is reduced to one value d2, then eliminate d2 from peers
    if len(values[s]) == 0:
        return False  # contradiction removed last value
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False

    # (2) --> if a unit u is reduced to only one place for a value d,then put it there.

    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False
        elif len(dplaces) == 1:
            # d can only be in 1 place in unit; assign it there
            if not assign(values, dplaces[0], d):
                return False
    return values


values = dict((s, digits) for s in squares)


def assign(values, s, d):
    """eliminate all other values(except d) from the values[s]"""
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False


# The parse_grid function
# the outcome of this parsing process leaves only possible values for each square
def parse_grid(grid):
    # given grid contains the initial representation of sudoku puzzle
    """convert grid to a dict of possible values, {square:digits}, or
    return false if a contradiction is detected"""
    # to start every square can be any digit, then assign values from the grid
    values = dict((s, digits) for s in squares)
    for s, d in grid_values(extract_grid_values(grid)).items():
        if str(d) in digits and not assign(values, s, str(d)):
            print("assign values", assign(values, s, str(d)))
            return False  # fail if we cant assign d to square s

    return values


def some(seq):
    "Return some element of seq that is true."
    for e in seq:
        if e:
            return e
    return False


def search(values):
    """using depth-first search and propagation, try all possible values"""
    if values is False:
        return False
    if all(len(values[s]) == 1 for s in squares):
        return values

    # chose the unfilled square s with the fewesr possibilities
    n, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    return some(search(assign(values.copy(), s, d)) for d in values[s])

def display(values):
    "Display these values as a 2-D grid."
    print("values to be displayed", values)
    width = 1+max(len(values[s]) for s in squares)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '') for c in cols))
        if r in 'CF':
            print(line)
    print()



