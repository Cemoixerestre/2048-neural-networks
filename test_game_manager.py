"""File to run with pytest"""

from game_manager import *

import pytest

@pytest.fixture(scope="module")
def board():
    """
    2  4  4 16
    .  2  .  2
    2  2  4  2
    2  .  .  ."""
    return np.array([[1, 2, 2, 4],
                     [0, 1, 0, 1],
                     [1, 1, 2, 1],
                     [1, 0, 0, 0]])



def test_is_not_over(board):
    assert not is_over(board)

def test_after_state_up(board):
    """
    4  4  8  16
    2  4  .  4
    .  .  .  .
    .  .  .  .

        /\
        ||
        
    2  4  4 16
    .  2  .  2
    2  2  4  2
    2  .  .  ."""
    afs, reward = after_state(board, (-1, 0))
    assert np.all(afs == np.array([[2, 2, 3, 4],
                                   [1, 2, 0, 2],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]])) and reward == 20

def test_after_state_left(board):
    """
    2  8  16 .      2  4  4 16  
    4  .  .  .  /—  .  2  .  2
    4  4  2  .  \—  2  2  4  2
    2  .  .  .      2  .  .  ."""
    afs, reward = after_state(board, (0, -1))
    assert np.all(afs == np.array([[1, 3, 4, 0],
                                   [2, 0, 0, 0],
                                   [2, 2, 1, 0],
                                   [1, 0, 0, 0]])) and reward == 16

def test_after_state_down(board):
    """
    2  4  4 16
    .  2  .  2
    2  2  4  2
    2  .  .  .
    
        ||
        \/

    .  .  .  .
    .  .  .  .
    2  4  .  16
    4  4  8  4"""
    afs, reward = after_state(board, (1, 0))
    assert np.all(afs == np.array([[0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [1, 2, 0, 4],
                                   [2, 2, 3, 2]])) and reward == 20

def test_after_state_right(board):
    """
    2  4  4 16      .  2  8  16
    .  2  .  2  —\  .  .  .  4
    2  2  4  2  —/  .  4  4  2
    2  .  .  .      .  .  .  2"""
    afs, reward = after_state(board, (0, 1))
    assert np.all(afs == np.array([[0, 1, 3, 4],
                                   [0, 0, 0, 2],
                                   [0, 2, 2, 1],
                                   [0, 0, 0, 1]])) and reward == 16
def test_is_over():
    assert is_over(np.array([[2, 1, 2, 3],
                             [4, 2, 3, 4],
                             [1, 3, 1, 2],
                             [3, 2, 3, 1]]))
