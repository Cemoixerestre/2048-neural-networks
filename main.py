import theano as th
from theano import tensor as T
from theano import sparse

import lasagne
from lasagne.layers import InputLayer, DenseLayer, get_output, get_all_params
from lasagne.nonlinearities import rectify, tanh
from lasagne.init import Normal, Constant
from lasagne.updates import sgd
from lasagne.utils import floatX, one_hot

from game_manager import *
from stats_manager import *

from time import time
from timeit import timeit

#m : each case can have m=18 values (between 0 and 131072)
m = 18
#the shape of the input :
shape = (1, 4*4*m)

#the learning rate :
alpha = 1e-5

board_var = T.lmatrix("board")
one_hot_repr = lasagne.utils.one_hot(board_var, m)
one_hot_repr = T.reshape(one_hot_repr, shape)
input_layer = InputLayer(shape, one_hot_repr, name="input-bis")
layer_1 = DenseLayer(input_layer, num_units=300, nonlinearity=rectify)
layer_2 = DenseLayer(layer_1, num_units=100, nonlinearity=rectify)
layer_3 = DenseLayer(layer_2, num_units=50, nonlinearity=tanh)
output_layer = DenseLayer(layer_3, num_units=1)
# We cast the output of the network into a scalar
output_layer = lasagne.layers.reshape(output_layer, ())

def TD0(network, board_var, nb_games, alpha, stats=StatsManager(100)):    #The output variable of the network :
    output = get_output(network)
    
    #The function evaluating a board :
    evaluate = th.function([board_var], output)
    
    #A symbolic variable denoting the expected output of a network :
    expected = T.scalar("expected output")
    cost = (output - expected) ** 2 / 2
    updates = sgd(cost, get_all_params(network), alpha)
    #perform a step of gradient descent :
    update = th.function([board_var, expected], updates=updates)

    print("compiled")

    def choose_next(board):
        """Explore the possible moves.
        Return a 3-uple (after_state, evaluation, reward) where
        -after_state with the best evaluation
        -evaluation is the evaluation of the network
        -reward is the reward after having applied the corresponding move."""
        l = []
        for move in all_moves:
            new, r = after_state(board, move)
            
            if np.all(board == new):
                #if no modification has been applied, this move is not legal.
                continue
            
            l.append((new, evaluate(new), r))
            
        return max(l, key=lambda x: x[1])

    for i in range(nb_games):
        #we always keep in memory :
        #- the previous_board s'_{t-1}, which is an after-state
        #- the board s_t, which is previous_board after having added a
        #  random tile.
        starting_board = new_board()
        previous_board, _, score = choose_next(starting_board)
        board = add_random_tile(previous_board)
        while not is_over(board):
            next_board, next_eval, reward = choose_next(board)
            update(previous_board, 1 + next_eval)
            previous_board = next_board
            board = add_random_tile(next_board)
            score += reward

        update(previous_board, 0)
        stats.update(board, score, network)

t = time()
stats = StatsManager(100)
TD0(output_layer, board_var, 10000, alpha, stats)
print(time() - t, "seconds")
