import numpy as np
from game_manager import display_board

class StatsManager:
    """Objects that record and display statistics about the training of a
    2048 network."""
    def __init__(self, display_time):
        """Each display_time games, we display, then reset statistics."""
        self.display_time = display_time
        self.nb_recording = 0
        #self.last_tile[i] is the number of times the highest tile at the
        #end of the game was i.
        self.last_tile = [0] * 18
        self.score_cum = 0

        self.history = []

    def update(self, final_board, score, network):
        """This function is called at the end of the game.
        It takes several arguments including the disposition of the final
        board, the score and the values of the parameters of the network."""
        self.nb_recording += 1
        self.last_tile[np.max(final_board)] += 1
        self.score_cum += score
        if self.nb_recording % self.display_time == 0:
            self.display(final_board, network)

    def display(self, final_board, network):
        """Display and reset the statistics."""
        print("Games:", self.nb_recording)
        print("Nb 256 :", sum(self.last_tile[8:]))
        print("Nb 512 :", sum(self.last_tile[9:]))
        print("Nb 1024:", sum(self.last_tile[10:]))
        print("Nb 2048:", sum(self.last_tile[11:]))
        print("Nb 4096:", sum(self.last_tile[12:]))
        print("Nb 8192:", sum(self.last_tile[13:]))
        print("Average Score:", self.score_cum / self.display_time)
        display_board(final_board)
        print()
        self.last_tile = [0] * 18

        self.history.append(([p.get_value() for p in network.get_params()],
                             self.score_cum / self.display_time))

        self.score_cum = 0
