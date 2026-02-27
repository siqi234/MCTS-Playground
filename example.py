import math
import random


import gymnasium as gym  



class Node:
    def __init__(self, parent=None, action=None, to_play=None):
        self.parent = parent
        self.action = action              
        self.children = {}                
        self.visits = 0
        self.value_sum = 0.0              
        self.untried_actions = [0, 1]     
        self.to_play = to_play            

    def uct(self, c=1.4):
        if self.visits == 0:
            return float("inf")
        exploit = self.value_sum / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore


def get_env_state(env):
    return env.unwrapped.state.copy()

def set_env_state(env, state):
    env.unwrapped.state = state.copy()

def rollout(env, depth_limit=50):
    total = 0.0
    for _ in range(depth_limit):
        a = random.choice([0, 1])
        obs, r, terminated, truncated, info = env.step(a)
        total += r
        if terminated or truncated:
            break
    return total

def mcts_iteration(env, root, depth_limit=50, c=1.4):
    saved = get_env_state(env)

    node = root

    while len(node.untried_actions) == 0 and node.children:
        node = max(node.children.values(), key=lambda ch: ch.uct(c))
        obs, r, terminated, truncated, info = env.step(node.action)
        if terminated or truncated:
            break

    if node.untried_actions:
        a = node.untried_actions.pop()
        obs, r, terminated, truncated, info = env.step(a)
        child = Node(parent=node, action=a)
        node.children[a] = child
        node = child

    value = 0.0
    if not (terminated or truncated):
        value = rollout(env, depth_limit=depth_limit)

    while node is not None:
        node.visits += 1
        node.value_sum += value
        node = node.parent

    set_env_state(env, saved)

def mcts_action(env, iterations=200, depth_limit=50, c=1.4):
    root = Node()

    for _ in range(iterations):
        mcts_iteration(env, root, depth_limit=depth_limit, c=c)

    best_action = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
    return best_action

def main():
    env = gym.make("CartPole-v1", render_mode="human")
    obs, info = env.reset(seed=0)
    total_reward = 0.0
    for t in range(500):
        a = mcts_action(env, iterations=300, depth_limit=60, c=1.4)

        obs, r, terminated, truncated, info = env.step(a)
        total_reward += r

        if terminated or truncated:
            break

    env.close()
    print("Episode reward:", total_reward)

if __name__ == "__main__":
    main() 