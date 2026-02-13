import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

# --- Tic-Tac-Toe state ---
# board: list of 9 ints: 0 empty, +1 X, -1 O
# player: +1 if X to move, -1 if O to move
@dataclass(frozen=True)
class TTTState:
    board: Tuple[int, ...] = field(default_factory=lambda: (0,)*9)
    player: int = +1            # +1 or -1

def legal_action(s: TTTState) -> List[int]:
    return [i for i, v in enumerate(s.board) if v==0 ]

def next_state(s: TTTState, a: int) -> TTTState:
    b = list(s.board)
    b[a] = s.player
    return TTTState(board=tuple(b), player=-s.player)

WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8), # horizontals
    (0,3,6),(1,4,7),(2,5,8), # verticals
    (0,4,8),(2,4,6) # diagonals
]

def winner(board: Tuple[int, ...]) -> int:
    # Return +1 if X wins, -1 if O wins, and 0 if no winner

    for (i, j, k) in WIN_LINES:
        s = board[i] + board[j] + board[k]
        if s == 3:
            return +1
        if s == -3:
            return -1
    return 0

def is_terminal(s: TTTState) -> bool:
    return winner(s.board) !=0 or all(v !=0 for v in s.board)

def terminal_value_from_X(s: TTTState) -> float:
    w = winner(s.board)
    if w == +1:
        return 1.0
    if w == -1:
        return -1.0
    return 0.0

# --- MCTS Node ---
@dataclass
class Node:
    state: TTTState
    parent: Optional['Node'] = None
    parent_action: Optional[int] = None
    children: Dict[int, 'Node'] = field(default_factory=dict)
    untried_actions: List[int] = field(default_factory=list)
    N: int = 0
    W: float = 0.0 

    def __post_init__(self):
        if not self.untried_actions:
            self.untried_actions = legal_action(self.state)

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0
    
def UCB1_score(parent: Node, child: Node, c: float = 1.414) -> float:
    if child.N == 0:
        return float("inf")

    explore = c * math.sqrt(math.log(parent.N + 1) / child.N)

    # Q is from X perspective everywhere (your Option B)
    # If it's X to move at parent, maximize Q
    # If it's O to move at parent, minimize Q  <=> maximize (-Q)
    exploit = child.Q if parent.state.player == +1 else -child.Q

    return exploit + explore


def select_child(node: Node, c: float) -> Node:
    # Maximize UCB1 score
    return max(
        node.children.values(), key=lambda ch: UCB1_score(node, ch, c)
    )

def expand(node: Node) -> Node: 
    a = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
    s2 = next_state(node.state, a)
    child = Node(state=s2, parent=node, parent_action=a)
    node.children[a] = child

    return child

def rollout(state: TTTState) -> float:
    s = state
    while not is_terminal(s):
        a = random.choice(legal_action(s))
        s = next_state(s, a)
    return terminal_value_from_X(s)

def backprop(node: Node, value_from_X: float):
    # Convert X-perspective value into each node's "player-to-move" perspective.
    # If node.state.player == +1 (X to move), value is as-is.
    # If node.state.player == -1 (O to move), value should be negated.
    curr = node
    while curr is not None:
        curr.N += 1
        curr.W += value_from_X
        curr = curr.parent

def mcts_search(root_state: TTTState, iters: int = 5000, c: float = 1.414) -> int:
    root = Node(state=root_state)

    for _ in range(iters):
        node = root

        # 1 Selection
        while node.untried_actions == [] and node.children:
            node = select_child(node, c)

        # 2 Expansion
        if node.untried_actions and not is_terminal(node.state):
            node = expand(node)

        # 3 Rollout
        value = rollout(node.state)

        # 4 Backpropagation
        backprop(node, value)

    best_action, best_child = max(root.children.items(), key=lambda kv: kv[1].N)

    print("MCTS results:")
    for a, ch in sorted(root.children.items(), key=lambda kv: -kv[1].N):
        print(a, "N", ch.N, "Q", round(ch.Q, 3))


    return best_action

# --- Pretty printing / demo ---
def render(board: Tuple[int, ...]) -> str:
    sym = {+1:"X", -1:"O", 0:"."}
    rows = []
    for r in range(3):
        rows.append(" ".join(sym[board[3*r + c]] for c in range(3)))
    return "\n".join(rows)

# --- Main Program ---
if __name__ == '__main__':
    # s = TTTState(board=tuple([0]*9), player=+1)

    s = TTTState(
        board=(
            -1,  0,  0,
            +1, -1,  0,
             0, +1,  0
        ),
        player=+1  # X to move must block at 2
    )

    a = mcts_search(s, iters=5000)
    print("Chosen action:", a)
    print(render(next_state(s, a).board))
