class QLearningAgent():
    # np.ndarray
    def __init__(self, sudoku_board):
        pass

    def reset(self):
        pass

    # Tuple[int, int, int]
    def step(self, action):
        pass

    def render(self):
        pass

    def is_valid_move(self, row: int, col: int, num: int):
        return False

    def is_solved(self):
        return False
    # (self, state: np.ndarray)
    
    def get_available_actions(self, bord_state):
        return None # List[Tuple[int, int, int]]