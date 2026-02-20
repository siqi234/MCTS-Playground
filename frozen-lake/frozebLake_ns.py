import gymnasium as gym
import math
import random
from graphviz import Digraph
import json
import os

# ----------------------------
# Utilities: action names
# ----------------------------
ACTION_MAP = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}


# ----------------------------
# Utilities: serialize + visualize tree
# ----------------------------
def serialize_tree(node, max_depth=3):
    """
    Convert the MCTS tree rooted at `node` to a dict.
    max_depth limits recursion depth to keep JSON small.
    """
    if node is None or max_depth < 0:
        return None

    win_rate = (node.value / node.visits) if node.visits > 0 else 0.0

    return {
        "state": int(node.state),
        "action_from_parent": None if node.action is None else int(node.action),
        "visits": int(node.visits),
        "total_value": float(node.value),
        "win_rate": round(float(win_rate), 4),
        "children": [serialize_tree(child, max_depth - 1) for child in node.children],
    }


def visualize_mcts_tree(root, filename="mcts_step", max_nodes=300):
    """
    Render a PNG of the current tree using Graphviz.

    - To keep things readable, we only plot nodes with >= min_visits_to_plot
      (auto-tuned based on root.visits).
    - max_nodes is a safety cap to avoid enormous graphs.
    """
    dot = Digraph(comment="MCTS Search Tree")
    dot.attr(rankdir="LR")  # left-to-right
    dot.attr("node", shape="box")

    # auto threshold: early steps draw more; later steps draw only meaningful nodes
    min_visits_to_plot = 1 if root.visits < 200 else 5
    if root.visits > 1000:
        min_visits_to_plot = 10

    dot.node("Root", label=f"State: {root.state}\nVisits: {root.visits}\nWR: {(root.value/root.visits) if root.visits else 0:.3f}")

    queue = [(root, "Root")]
    idx = 0
    drawn = 1

    while queue and drawn < max_nodes:
        current_node, parent_id = queue.pop(0)

        for child in current_node.children:
            if child.visits < min_visits_to_plot:
                continue

            idx += 1
            child_id = f"node_{idx}"

            move = ACTION_MAP.get(child.action, "?")
            win_rate = (child.value / child.visits) if child.visits > 0 else 0.0

            label = (
                f"Move: {move}\n"
                f"State: {child.state}\n"
                f"WR: {win_rate:.3f}\n"
                f"N: {child.visits}"
            )

            # Simple color coding
            color = "green" if win_rate >= 0.5 else "red"

            dot.node(child_id, label=label, color=color)
            dot.edge(parent_id, child_id)

            queue.append((child, child_id))
            drawn += 1
            if drawn >= max_nodes:
                break

    dot.render(filename, format="png", cleanup=True)
    print(f"Tree saved to {filename}.png (min_visits_to_plot={min_visits_to_plot}, nodes={drawn})")


# ----------------------------
# MCTS Implementation (stochastic transitions via env.unwrapped.P)
# ----------------------------
EXPLORATION_CONSTANT = math.sqrt(2)


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = int(state)
        self.parent = parent
        self.action = action  # action taken to reach this node (None for root)
        self.children = []

        self.visits = 0
        self.value = 0.0  # total return (sum of rollout rewards)

        # In FrozenLake actions are always 0..3
        self.untried_actions = [0, 1, 2, 3]

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=EXPLORATION_CONSTANT):
        """
        Standard UCB1:
          (Q/N) + c * sqrt(2*ln(N_parent)/N_child)
        """
        best = None
        best_score = -1e18

        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = child.value / child.visits
                explore = c_param * math.sqrt((2.0 * math.log(self.visits)) / child.visits)
                score = exploit + explore

            if score > best_score:
                best_score = score
                best = child

        return best


class MCTS:
    def __init__(self, env, max_rollout_steps=30):
        self.env = env
        # Transition model: P[s][a] = [(prob, next_state, reward, done), ...]
        self.P = env.unwrapped.P
        self.max_rollout_steps = max_rollout_steps

        # Pre-compute terminal states from the map (robust, no hardcoding holes)
        desc = env.unwrapped.desc  # bytes array
        n = desc.shape[0]
        holes = set()
        goal = None

        for r in range(n):
            for c in range(n):
                ch = desc[r, c].decode("utf-8")
                s = r * n + c
                if ch == "H":
                    holes.add(s)
                elif ch == "G":
                    goal = s

        self.holes = holes
        self.goal = goal

    def search(self, initial_state, iterations=1000):
        root = MCTSNode(state=initial_state)

        for _ in range(iterations):
            node = self._select(root)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        return self._get_best_action(root), root

    def _select(self, node):
        """
        Selection: follow best_child until we find a node to expand or terminal.
        """
        while not self._is_terminal(node.state):
            if not node.is_fully_expanded():
                return self._expand(node)
            node = node.best_child()
        return node

    def _expand(self, node):
        """
        Expansion: take one untried action, sample a stochastic next_state (chance),
        create a new child.
        """
        action = node.untried_actions.pop()  # pop one action
        next_state = self._get_next_state(node.state, action)  # chance happens here
        child_node = MCTSNode(state=next_state, parent=node, action=action)
        node.children.append(child_node)
        return child_node

    def _simulate(self, state):
        """
        Rollout: random actions until terminal or max steps.
        Return 1.0 if goal reached else 0.0.
        """
        current_state = int(state)
        steps = 0

        while (not self._is_terminal(current_state)) and steps < self.max_rollout_steps:
            action = random.choice([0, 1, 2, 3])
            current_state = self._get_next_state(current_state, action)  # chance
            steps += 1

        return 1.0 if current_state == self.goal else 0.0

    def _backpropagate(self, node, reward):
        """
        Backprop: add reward to all ancestors.
        """
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _get_next_state(self, state, action):
        """
        Chance node sampling using the true transition model.
        """
        transitions = self.P[state][action]
        probs = [t[0] for t in transitions]
        states = [t[1] for t in transitions]
        return random.choices(states, weights=probs, k=1)[0]

    def _is_terminal(self, state):
        return (state in self.holes) or (state == self.goal)

    def _get_best_action(self, root):
        """
        Robust child: most visited.
        """
        if not root.children:
            return random.choice([0, 1, 2, 3])
        return max(root.children, key=lambda c: c.visits).action


# ----------------------------
# Main Game Loop
# ----------------------------
if __name__ == "__main__":
    # Output folders
    out_dir = "frozen-lake/slippery_tree"
    os.makedirs(out_dir, exist_ok=True)

    # Setup environment
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human")
    observation, info = env.reset()

    mcts = MCTS(env, max_rollout_steps=30)

    history = []
    done = False
    step_count = 0

    print("Start MCTS Agent on *Slippery* Frozen Lake...")

    while not done and step_count < 200:
        # SEARCH: Run MCTS on current state
        best_action, tree_root = mcts.search(initial_state=observation, iterations=1000)

        # Record this step
        step_data = {
            "step": step_count,
            "current_state": int(observation),
            "chosen_action": int(best_action),
            "chosen_action_name": ACTION_MAP.get(best_action, "?"),
            "tree_snapshot": serialize_tree(tree_root, max_depth=3),
            "root_visits": int(tree_root.visits),
            "root_win_rate": float(tree_root.value / tree_root.visits) if tree_root.visits else 0.0,
        }
        history.append(step_data)

        # Visualize tree
        visualize_mcts_tree(tree_root, filename=os.path.join(out_dir, f"mcts_step_{step_count:03d}"))

        # ACT: Apply action to real env (slippery randomness happens here too)
        observation, reward, terminated, truncated, info = env.step(best_action)
        done = terminated or truncated

        step_count += 1

        if done:
            if reward == 1:
                print("Goal Reached!")
            else:
                print("Fell in a hole.")

    # Save history JSON
    json_path = os.path.join(out_dir, "slippery_run_history.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print("Saved history to:", json_path)
    env.close()