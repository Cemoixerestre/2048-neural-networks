import numpy as np
from random import choice, random
from itertools import product

"""Module game_manager:
[WRITEME]

A 2048 board is represented as a 4*4 numpy int array.
Let b be a board. For each 0 ≤ x, y < 4, b[x, y] represents the case at line x
and column y. The case value is 0 if the case is empty, p if the case value
is 2^p."""

#the moves are reprensented by a a direction among :
all_moves = [(0, 1), (1, 0), (-1, 0), (0, -1)]
#(0, 1) means right, (1, 0) means up, etc...

def new_board():
    """Create a new 2048 board.

    All the cases are empty, except 2."""
    b = np.zeros((4, 4), dtype="int8")
    b = add_random_tile(b)
    b = add_random_tile(b)
    return b

def add_random_tile(board):
    empty_cases = [c for c in product(range(4), repeat=2) if board[c] == 0]
    case = choice(empty_cases) #if there is no empty cases, raises the
                               #appropriate exception.
    new_board = board.copy()
    new_board[case] = 1 if random() <= 0.9 else 2
    return new_board

def shift(row):
    """Shift a row to the left. Returns the new row and the gain.
    row: a numpy int array of size 4.

    Ex : shift([1, 1, 0, 2]) = [2, 2, 0, 0]"""
    new_row = np.zeros(4, int)
    dest = 0
    gain = 0
    for i in range(4):
        if row[i] == 0:
            continue
        if new_row[dest] == 0:
            #la case se déplace vers une case vide
            new_row[dest] = row[i]
        elif new_row[dest] == row[i]:
            #fusion
            gain += 2 ** (row[i] + 1)
            new_row[dest] += 1
            dest += 1
        else:
            dest += 1
            new_row[dest] = row[i]

    return new_row, gain

def after_state(board, direction):
    """apply a move to a 2048 board. Return the after-state and the reward.

    board: a numpy int array of size (4, 4)
           it represents the 2048 board."""
    new_board = np.ndarray((4, 4), int)
    gain = 0
    if direction == (0, -1): #left
        for i in range(4):
            new_row, rgain = shift(board[i, :])
            new_board[i, :] = new_row
            gain += rgain
    elif direction == (-1, 0): #down
        for i in range(4):
            new_row, rgain = shift(board[:, i])
            new_board[:, i] = new_row
            gain += rgain
    elif direction == (0, 1): #right
        for i in range(4):
            new_row, rgain = shift(board[i, ::-1])
            new_board[i, ::-1] = new_row
            gain += rgain
    else:
        for i in range(4): #up
            new_row, rgain = shift(board[::-1, i])
            new_board[::-1, i] = new_row
            gain += rgain
            
    return new_board, gain

def is_over(board):
    """Return True is no move is available."""
    if 0 in board:
        #there is an empty case
        return False
    
    for i in range(3):
        for j in range(4):
            if board[i, j] == board[i+1, j] or \
               board[j, i] == board[j, i+1]:
                #these two tiles can merge :
                return False

    return True

def move(board, direction):
    afs = after_state(board, direction)[0]
    return add_random_tile(afs)

def available_moves(board):
    """Renvoie une l de booléens à 4 éléments telle que l[i] est True si et
    seulement si le coup all_moves[i] est valable."""
    res = []
    for move in all_moves:
        available = np.any(board != apply_move(board, move)[0])
        res.append(available)
    return res

def display_board(board):
    for line in board:
        for x in line:
            r = 0 if x == 0 else 2**x
            print("%6d" % r, end="")
        print()
