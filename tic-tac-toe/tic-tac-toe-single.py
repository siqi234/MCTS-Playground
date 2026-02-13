import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import json
from collections import deque


# --- Tic-Tac-Toe state ---
# board: list of 9 ints: 0 empty, +1 X, -1 O
# player: +1 if X to move, -1 if O to move
@dataclass(frozen=True)
class TTTState:
    '''
        Create the game state for TTT
    '''
    board: Tuple[int, ...] = field(default_factory=lambda: (0,)*9)
    player: int = +1            # +1 or -1

def legal_action(s: TTTState) -> List[int]:
    ''' 
        Return list of legal action indices (0-8) for the given state 
    '''
    return [i for i, v in enumerate(s.board) if v==0 ]

def next_state(s: TTTState, a: int) -> TTTState:
    '''
        Return the next state after taking action a in state s.
    '''
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
    
def UCB1_score(parent: Node, child: Node, c: float = math.sqrt(2)) -> float:
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


    return best_action, root

# --- Pretty printing / demo ---
def render(board: Tuple[int, ...]) -> str:
    sym = {+1:"X", -1:"O", 0:"."}
    rows = []
    for r in range(3):
        rows.append(" ".join(sym[board[3*r + c]] for c in range(3)))
    return "\n".join(rows)

# Visualization of the tree
def export_tree_json(root: Node, path: str, max_nodes: int = 5000) -> None:
    """
    Export up to max_nodes nodes from the MCTS tree rooted at `root` to a JSON file.
    Each node stores state, stats, parent info, and children ids.
    """
    # Assign ids as we discover nodes
    node_to_id = {id(root): 0}
    nodes = []
    q = deque([root])

    def state_dict(s: TTTState) -> dict:
        return {
            "board": list(s.board),
            "player": s.player
        }

    while q and len(nodes) < max_nodes:
        n = q.popleft()
        nid = node_to_id[id(n)]

        # Map children actions to child ids
        child_map = {}
        for a, ch in n.children.items():
            if id(ch) not in node_to_id:
                node_to_id[id(ch)] = len(node_to_id)
                q.append(ch)
            child_map[a] = node_to_id[id(ch)]

        nodes.append({
            "id": nid,
            "parent_id": node_to_id[id(n.parent)] if n.parent is not None else None,
            "parent_action": n.parent_action,
            "state": state_dict(n.state),
            "N": n.N,
            "W": n.W,
            "Q": (n.W / n.N) if n.N > 0 else 0.0,
            "untried_actions": list(n.untried_actions),
            "children": child_map
        })

    out = {
        "root_id": 0,
        "max_nodes": max_nodes,
        "exported_nodes": len(nodes),
        "nodes": nodes
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

# Visualize the whole tree
def print_tree(node: Node, depth: int = 2, top_k: int = 3, indent: str = ""):
    if depth < 0:
        return
    # sort children by visit count
    kids = sorted(node.children.items(), key=lambda kv: -kv[1].N)[:top_k]
    for a, ch in kids:
        print(f"{indent}- a={a} N={ch.N} Q={ch.Q:.3f} player_to_move={ch.state.player}")
        print_tree(ch, depth=depth-1, top_k=top_k, indent=indent + "  ")

def export_tree_dot(root: Node, path: str, max_nodes: int = 500):
    from collections import deque
    q = deque([root])
    node_to_id = {id(root): 0}
    lines = ["digraph MCTS {", "  node [shape=box];"]

    def label(n: Node) -> str:
        b = "".join({1:"X",-1:"O",0:"."}[v] for v in n.state.board)
        return f"id={node_to_id[id(n)]}\\nN={n.N} Q={n.Q:.3f}\\nP={n.state.player}\\n{b}"

    exported = 0
    while q and exported < max_nodes:
        n = q.popleft()
        nid = node_to_id[id(n)]
        lines.append(f'  {nid} [label="{label(n)}"];')

        for a, ch in n.children.items():
            if id(ch) not in node_to_id:
                node_to_id[id(ch)] = len(node_to_id)
                q.append(ch)
            cid = node_to_id[id(ch)]
            lines.append(f"  {nid} -> {cid} [label=\"a={a}\"];")

        exported += 1

    lines.append("}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



# --- Main Program ---
if __name__ == '__main__':
    # s = TTTState(board=tuple([0]*9), player=+1)

    # Create board game
    # s = TTTState(
    #     board=(
    #         -1,  0,  0,
    #         +1, -1,  0,
    #          0, +1,  0
    #     ),
    #     player=+1  # X to move must block at 8
    # )

    s = TTTState(
        board=(
            -1,  0,  0,
            +1, +1,  0,
             0, -1,  0
        ),
        player=+1  # Win at 5
    )
    
    print(render(s.board))


    # Run MCTS
    a, root = mcts_search(s, iters=5000)
    # export_tree_json(root, "mcts_tree_block.json", max_nodes=5000)
    # export_tree_dot(root, "mcts_tree_block.dot", max_nodes=5000)
    # print_tree(root, depth=3, top_k=5)

    print("Chosen action:", a)
    print(render(next_state(s, a).board))


    # # # test legal actions
    # legal_actions = legal_action(s)
    # print("Legal actions:", legal_actions)

    # # test next state
    # a = 2
    # s2 = next_state(s, a)
    # print("Next state after action", a)
    # print(render(s2.board))





