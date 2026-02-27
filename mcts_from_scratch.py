import gymnasium as gym
import math
import random

class DecisionNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

        self.children = {} # {action_id: ChanceNode}, where action_id is the action taken from this desision node
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self, action_space_size):
        # A decision node is fully expanded if it has children for all possible actions
        return len(self.children) == action_space_size
    
    def best_child(self, c_param=math.sqrt(2)):
        best_score = float('-inf')
        best_child = None

        # iterate through all children (chance nodes) and calculate the UCB score for each
        for action_id, chance_node in self.children.items():

            if chance_node.visits == 0:
                return action_id
            
            exploit = chance_node.value / chance_node.visits
            explore  = c_param * math.sqrt(math.log(self.visits)/(chance_node.visits))
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_child = action_id
        
        return best_child
    
class ChanceNode:
    def __init__(self, parent, action_id):
        self.state = parent.state # from the parent decision node
        self.parent = parent # should be a decision node
        self.action_id = action_id 

        self.children = {} # {next_state: DecisionNode}
        self.visits = 0
        self.value = 0.0


# MCTS
class MCTS:
    # Initialize the MCTS with the environment
    def __init__(self, env, iterations=1000):
        self.env = env
        self.iterations = iterations
        self.action_space_size = env.action_space.n

    def is_terminal(self, state):
        if state == 15: 
            return True
        holes = {5, 7, 11, 12} 
        return state in holes or state == 15

    def search(self, initial_state):
        root = DecisionNode(state=initial_state)

        for _ in range(self.iterations):
            node = self._select(root)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)
        
        return root.best_child(c_param=0)

    def _select(self, node):
        # Selection
        while not self.is_terminal(node.state):
            if not node.is_fully_expanded(self.action_space_size):
                return self._expand(node)
            
            best_action = node.best_child()
            chance_node = node.children[best_action]

            self.set_env_state(node.state)

            # Sample next state from the chance node's distribution
            next_state, reward, done, _, _ = self.env.step(best_action)
            
            if next_state not in chance_node.children:
                chance_node.children[next_state] = DecisionNode(state=next_state, parent=chance_node)

            node = chance_node.children[next_state]

            if done:
                break

        return node
    
    def _expand(self, node):
        # Expansion
        child_node = None
        tried_action = node.children.keys()
        untried_actions = [a for a in range(self.action_space_size) if a not in tried_action]

        action = random.choice(untried_actions)

        chance_node = ChanceNode(parent=node, action_id = action)
        node.children[action] = chance_node

        self.set_env_state(node.state)

        next_state, reward, done, _, _ = self.env.step(action)

        next_node = DecisionNode(state=next_state, parent=chance_node)
        chance_node.children[next_state] = next_node
        child_node = next_node

        return child_node
   
    def _simulate(self, state):
        current_state = state
        self.set_env_state(current_state)
        done = self.is_terminal(current_state)

        if done:
            return 1.0 if current_state == 15 else -1.0
        
        total_rewards = 0.0
        depth = 0
        max_depth = 50

        while not done and depth < max_depth:
            action = self.env.action_space.sample()
            next_state, reward, done, truncated, _ = self.env.step(action)

            if done:
                if next_state == 15:
                    total_rewards += 1.0
                else:
                    total_rewards += -1.0
                break
            
            total_rewards -= 0.01
            depth += 1
            current_state = next_state

            if truncated:
                break

        return total_rewards

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def set_env_state(self, state):
        self.env.unwrapped.s = state

if __name__ == "__main__":
    real_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode="human")

    sim_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    obs, info = real_env.reset(seed=42)
    sim_env.reset(seed=42)

    mcts = MCTS(sim_env, iterations=800)
    
    done = False
    step = 0
    truncated = False

    print("Start MCTS Agent on *Slippery* Frozen Lake...")
    while not (done or truncated):
        action = mcts.search(obs)
        obs, reward, done, truncated, info = real_env.step(action)
        step += 1

        if done:
            if reward == 1:
                print("Goal Reached!")
            else:
                print("Fell in a hole.")

    
    print(f"Episode finished after {step} steps with reward {reward}")
    real_env.close()
    sim_env.close()