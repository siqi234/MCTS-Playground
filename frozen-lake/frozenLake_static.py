import gymnasium as gym
import math
import random
import copy
from graphviz import Digraph
import json

# --- Utilities for visualization ---
def serialize_tree(node, max_depth=3):
    """
    Converts the MCTS tree into a dictionary.
    max_depth: Limits how deep we record to save space/time.
    """
    if node is None or max_depth < 0:
        return None

    # Calculate value for this node (e.g., win rate)
    win_rate = node.value / node.visits if node.visits > 0 else 0

    return {
        "state": int(node.state),       # The FrozenLake grid position (0-15)
        "action_from_parent": node.action, # The move taken to get here
        "visits": node.visits,
        "total_value": node.value,
        "win_rate": round(win_rate, 4),
        # Recursively serialize children
        "children": [
            serialize_tree(child, max_depth - 1) 
            for child in node.children
        ]
    }

def visualize_mcts_tree(root, filename="mcts_step"):
    dot = Digraph(comment='MCTS Search Tree')
    
    # We use a queue for Breadth-First Search to add nodes to the graph
    queue = [(root, "Root")]
    
    # To keep the graph readable, we might only plot nodes with > N visits
    min_visits_to_plot = 10 
    
    # Create the root node in the graph
    dot.node("Root", label=f"State: {root.state}\nVisits: {root.visits}", shape="box")

    idx = 0
    while queue:
        current_node, parent_id = queue.pop(0)
        
        for child in current_node.children:
            if child.visits >= min_visits_to_plot:
                idx += 1
                child_id = f"node_{idx}"
                
                # Label: Action taken + Win Rate
                action_map = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
                move = action_map.get(child.action, "?")
                win_rate = child.value / child.visits
                
                label = f"Move: {move}\nState: {child.state}\nWR: {win_rate:.2f}\nN: {child.visits}"
                
                # Color code: Green for high win rate, Red for low
                # Find the best sibling's win rate (e.g., 0.019)
                if win_rate > 0.5: color = "green"
                else: color = "red" # 1.9% will never be green
                
                dot.node(child_id, label=label, color=color)
                dot.edge(parent_id, child_id)
                
                queue.append((child, child_id))
    
    dot.render(filename, format='png', cleanup=True)
    print(f"Tree saved to {filename}.png")

# --- MCTS Implementation ---
# Standard UCB1 exploration constant
EXPLORATION_CONSTANT = math.sqrt(2)

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # The action taken to reach this node
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = [0, 1, 2, 3] # Left, Down, Right, Up

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=EXPLORATION_CONSTANT):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

class MCTS:
    def __init__(self, env):
        self.env = env
        # FrozenLake exposes the transition model P[state][action] = [(prob, next_state, reward, terminated)]
        self.P = env.unwrapped.P 

    def search(self, initial_state, iterations=1000):
        root = MCTSNode(state=initial_state)

        for _ in range(iterations):
            node = self._select(root)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        # Return the action of the most visited child
        return self._get_best_action(root), root

    def _select(self, node):
        while not self._is_terminal(node.state):
            if not node.is_fully_expanded():
                return self._expand(node)
            node = node.best_child()
        return node

    def _expand(self, node):
        action = node.untried_actions.pop()
        next_state = self._get_next_state(node.state, action)
        child_node = MCTSNode(state=next_state, parent=node, action=action)
        node.children.append(child_node)
        return child_node

    def _simulate(self, state):
        current_state = state
        # Run a random rollout until terminal state
        steps = 0
        while not self._is_terminal(current_state) and steps < 20:
            action = random.choice([0, 1, 2, 3])
            current_state = self._get_next_state(current_state, action)
            steps += 1
            
        # Return 1 if goal reached, 0 otherwise
        # In FrozenLake, state 15 is usually the goal (bottom right)
        return 1.0 if current_state == 15 else 0.0

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _get_next_state(self, state, action):
            # Fetch transitions from the environment's model
            # transitions = [(probability, next_state, reward, terminated), ...]
            transitions = self.P[state][action]
            
            # Extract probabilities and corresponding states
            probs = [t[0] for t in transitions]
            states = [t[1] for t in transitions]
            
            # Randomly select the next state based on the probabilities
            # This simulates the "Slippery" nature of the ice
            return random.choices(states, weights=probs)[0]

    def _is_terminal(self, state):
        # In 4x4 FrozenLake: Holes (H) and Goal (G) are terminal
        # We check if there are no transitions or if we are at the goal
        # (This logic depends on specific map, but standard map works like this)
        # Easier check: Is it the goal (15) or a hole?
        # For generality, we can check if the game ends in the env logic, 
        # but here we hardcode 15 for the standard map goal.
        if state == 15: return True
        
        # Check if it is a hole (reward 0 and terminal)
        # We can peek at any action to see if it leads to 'done' immediately with 0 reward
        # But simpler: just run the simulation loop carefully.
        # For this snippet, we treat 15 as the only positive terminal.
        # If we fall in a hole, the simulation loop will continue or we can detect it.
        # Let's assume standard map layout holes:
        holes = {5, 7, 11, 12} 
        return state in holes or state == 15

    def _get_best_action(self, root):
        # Exploitation: choose the child with the most visits (robustness)
        if not root.children:
            return random.choice([0, 1, 2, 3])
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        return sorted_children[0].action

# --- Main Game Loop ---
if __name__ == "__main__":
    # Setup environment
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human") # is_slippery=False for deterministic problems
    observation, info = env.reset()

    mcts = MCTS(env)
    # Storage for the game history
    history = []

    observation, info = env.reset()
    done = False
    step_count = 0

    print("Start MCTS Agent on *Non-Slippery* Frozen Lake...")

    while not done:
        # SEARCH: Run MCTS to find the best action
        # Note: We return only the action here (standard usage)
        # If you are using the 'recorder' version, change this to: action, root = ...
        best_action, tree_root = mcts.search(initial_state=observation, iterations=1000)

        # Record the step's data for visualization
        step_data = {
            "step": step_count,
            "current_state": int(observation),
            "chosen_action": int(best_action),
            "tree_snapshot": serialize_tree(tree_root, max_depth=3) 
        }
        history.append(step_data)
        visualize_mcts_tree(tree_root, filename=f"frozen-lake/non_slippry_tree/mcts_step_{step_count}")

        # ACT: Take the action in the real environment
        observation, reward, terminated, truncated, info = env.step(best_action)
        done = terminated or truncated

        step_count += 1

        if done:
            if reward == 1:
                print("Goal Reached!")
            else:
                print("Fell in a hole.")

    # Save the history to a JSON file
    with open("frozen-lake/non_slippry_tree/non_slippery_run_history.json", "w") as f:
        json.dump(history, f, indent=4)

    env.close()