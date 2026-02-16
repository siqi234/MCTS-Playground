import gymnasium as gym
import math
import random
import copy

# Standard UCB1 exploration constant
EXPLORATION_CONSTANT = 1.41

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
        return self._get_best_action(root)

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
        # We use the environment's transition matrix P to simulate the move.
        # Since we use is_slippery=False, the list has only 1 item (prob=1.0).
        transitions = self.P[state][action]
        # transitions is a list of tuples: (probability, new_state, reward, terminated)
        return transitions[0][1]

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

# Setup environment (Deterministic for basic MCTS)
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")
observation, info = env.reset()

mcts = MCTS(env)

print("Start MCTS Agent on Frozen Lake...")

done = False
while not done:
    # Run MCTS to find the best action for the current state
    action = mcts.search(initial_state=observation, iterations=1000)
    
    # Take the action in the real environment
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
        if reward == 1:
            print("Goal Reached!")
        else:
            print("Fell in a hole.")

env.close()