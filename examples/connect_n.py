import numpy as np
from scipy import signal

class ConnectN():

    def __init__(self, string_notation=None, size=(6,7), num_of_players:int=2, connect=4):
        
        self._board = np.zeros(size, dtype=int)
        self.num_of_players = num_of_players
        self.size = np.array(size)

        self.current_player = 1

        self.patterns = [
            np.ones((connect,1)),       # Vertical
            np.ones((1,connect)),       # Horizontal
            np.eye(connect),            # Diagonal 1
            np.flip(np.eye(connect),0), # Diagonal 2
        ]

        self.winner = -1
        self.is_game_over = False

        self.actions = np.arange(self.size[1], dtype=int)

        if string_notation != None:
            self.size, self.num_of_players, self.current_player, self._board = self._string_notation_to_state(string_notation)
    
    def __str__(self):
        return str(self._board)

    @property
    def board(self):
        return self._board.copy()

    def copy(self):
        game = ConnectN(size=self.size, num_of_players=self.num_of_players)
        game.patterns = [pattern.copy() for pattern in self.patterns]
        game.set_player(self.current_player)
        return game

    def set_player(self, player:int):
        self.current_player = player
        
    def add_pattern(self, pattern):
        self.patterns.append(pattern)

    def string_notation(self) -> str:
        size_notation = f"x{self.size[0]}y{self.size[1]}"
        num_player_notation = f"p{self.num_of_players}"
        current_player_notation = f"c{self.current_player}"
        board_notation = "b"+" ".join([str(v) for v in self._board.reshape(-1)])
        return size_notation + num_player_notation + current_player_notation + board_notation

    def _string_notation_to_state(self, notation: str):
        start_x = notation.find("x")
        start_y = notation.find("y")
        start_num_players = notation.find("p")
        startcurrent_player = notation.find("c")
        start_board = notation.find("b")
        
        x = notation[start_x+1:start_y]
        y = notation[start_y+1:start_num_players]
        num_players = int(notation[start_num_players+1:startcurrent_player])
        current_player = int(notation[startcurrent_player+1:start_board])
        size = np.array((x, y), dtype=int)
        pieces = notation[start_board+1:].split(" ")
        board = np.array(pieces,dtype=int).reshape(size)
        return size, num_players, current_player, board

    def step(self, column_index:int):
        # Check if the game is already over
        if self.winner != -1:
            raise RuntimeError("Game is already over")
        # Get the column of interest
        column = self._board[:,column_index]
        # check where there are already pieces
        got_piece = column > 0
        # Check if column is already full
        assert np.sum(got_piece) < self.size[1]; "This column is full"
        # Get index of the lowest empty spot
        row_index = np.where(got_piece==0)[0][-1]
        # Set piece
        self._board[row_index, column_index] = self.current_player
        # Set next player
        self.next_player()
        # Check if there is a winner
        self.winner = self.get_winner()
        if self.winner != -1:
            self.is_game_over = True
        # Check if the board is full
        if np.sum(self._board == 0) == 0:
            self.is_game_over = True

    def next_player(self):
        self.current_player += 1
        if self.current_player > self.num_of_players:
            self.current_player = 1

    def get_winner(self):
        for player in range(1, self.num_of_players+1):
            player_board = self._board == player
            for pattern in self.patterns:
                value = np.sum(pattern)
                match = signal.convolve2d(player_board, pattern)
                if np.sum(match == value) > 0:
                    return player
        return -1

    def get_legal_actions(self):
        return self.actions[self._board[0,:]==0]

    def get_legal_random_action(self):
        return np.random.choice(self.get_legal_actions())


if __name__ == "__main__":

    game = ConnectN()
    while game.is_game_over == False:
        game.step(game.get_legal_random_action())
    print(game._board)
    print(game.winner)

    game2 = ConnectN(string_notation=game.string_notation())