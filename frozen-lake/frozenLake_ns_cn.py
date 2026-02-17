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

    base = {
        "node_type": node.node_type,
        "state": int(node.state),
        "action_from_parent": node.action,
        "prob_from_parent": round(float(getattr(node, "prob", 1.0)), 4),
        "visits": int(node.visits),
        "total_value": float(node.value),
        "win_rate": round(win_rate, 4),
        "children": []
    }

    # chance nodes
    if node.node_type == "chance" and node.transitions is not None:
        base["transitions"] = [
            {"p": float(p), "next_state": int(ns), "reward": float(r), "terminated": bool(t)}
            for (p, ns, r, t) in node.transitions
        ]
    base["children"] = [serialize_tree(c, max_depth - 1) for c in node.children]

    return base


def visualize_mcts_tree(root, filename="mcts_step"):
    dot = Digraph(comment='MCTS Search Tree')
    
    # for Breadth-First Search to add nodes to the graph
    queue = [(root, "Root")]
    
    # plot nodes with > N visits for readability
    min_visits_to_plot = 0 # originally 10
    
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
                win_rate = child.value / child.visits
                
                if child.node_type == "chance":
                    move = action_map.get(child.action, "?")
                    label = f"CHANCE\nA: {move}\nN: {child.visits}\nWR: {win_rate:.2f}"
                else:
                    p = getattr(child, "prob", 1.0)
                    label = f"DECISION\nS: {child.state}\np: {p:.2f}\nN: {child.visits}\nWR: {win_rate:.2f}"

                
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
    def __init__(self, state, parent=None, action=None, node_type='decision', prob=1.0, transitions=None):
        self.state = state
        self.parent = parent
        self.action = action  # action taken to reach this node
        self.node_type = node_type # 'chance' or 'decision'

        self.children = []
        self.visits = 0
        self.value = 0.0

        self.untried_actions = [0, 1, 2, 3] if node_type == 'decision' else [] # Left, Down, Right, Up

        # transitions for chance nodes: list of (probability, next_state, reward, terminated)
        self.prob = prob
        self.transitions = transitions
        self.untried_outcomes = list(transitions) if (node_type == 'chance' and transitions is not None) else []

    def is_fully_expanded(self):
        return (len(self.untried_actions) == 0) if self.node_type == "decision" else (len(self.untried_outcomes) == 0)

    def best_child(self, c_param=EXPLORATION_CONSTANT):
        choices_weights = []

        for child in self.children:
            if child.visits == 0:
                score = float('inf')
            else:
                score  = (child.value / child.visits) +  c_param * math.sqrt(2*math.log(self.visits)/child.visits)
            choices_weights.append(score)

        return self.children[choices_weights.index(max(choices_weights))]

class MCTS:
    def __init__(self, env):
        self.env = env
        # FrozenLake exposes the transition model P[state][action] = [(prob, next_state, reward, terminated)]
        self.P = env.unwrapped.P 
        self.desc = env.unwrapped.desc.astype("U1")  # chars: b'S','F','H','G'
        self.nrow, self.ncol = self.desc.shape


    def search(self, initial_state, iterations=1000):
        root = MCTSNode(state=initial_state)

        print("DEBUG root type:", root.node_type)
        print("DEBUG root untried_actions:", root.untried_actions)
        print("DEBUG root fully_expanded?:", root.is_fully_expanded())

        # If selection ends on a chance node, move to an outcome node before rollout
        if root.node_type == "chance":
            # ensure at least one outcome child exists
            if not root.children and root.untried_outcomes:
                root = self._expand(root)
            elif root.children:
                probs = [c.prob for c in root.children]
                root = random.choices(root.children, weights=probs)[0]
            # now root should be a decision node at next_state
            
            
        reward = self._simulate(root.state)
        self._backpropagate(root, reward)

        # Return the action of the most visited child
        return self._get_best_action(root), root

    def _select(self, node):
        while True:
            # terminal check: for decision nodes, use state; for chance nodes, also use state
            if self._is_terminal(node.state):
                return node

            # If node not fully expanded, expand it
            if not node.is_fully_expanded():
                return self._expand(node)   # MUST return a node (never None)

            # Otherwise descend
            if node.node_type == "decision":
                # If fully expanded but no children, fallback expand (shouldn't happen, but prevents None)
                if not node.children:
                    return self._expand(node)
                node = node.best_child()
            else:
                # chance node: if no children yet, expand one outcome
                if not node.children and node.untried_outcomes:
                    return self._expand(node)

                # if still no children, something is wrong, but avoid None
                if not node.children:
                    return node

                probs = [c.prob for c in node.children]
                node = random.choices(node.children, weights=probs)[0]

    def _expand(self, node):
        if node.node_type == "decision":
            # expand by creating a chance node for an untried action
            if not node.untried_actions:
                return node  # safety

            action = node.untried_actions.pop()
            transitions = self.P[node.state][action]  # [(p, ns, r, term), ...]

            chance_node = MCTSNode(
                state=node.state,
                parent=node,
                action=action,
                node_type="chance",
                transitions=transitions
            )
            node.children.append(chance_node)
            return chance_node

        else:
            # expand by adding one stochastic outcome child
            if not node.untried_outcomes:
                return node  # safety

            p, next_state, r, terminated = node.untried_outcomes.pop()

            outcome_node = MCTSNode(
                state=next_state,
                parent=node,
                action=None,
                node_type="decision",
                prob=p
            )
            node.children.append(outcome_node)
            return outcome_node


    def _simulate(self, state):
        current_state = state
        # Run a random rollout until terminal state
        steps = 0

        while not self._is_terminal(current_state) and steps < 20:
            action = random.randrange(self.env.action_space.n)
            current_state, r, term = self._get_next_state(current_state, action)
            steps += 1
            if term:
                return float(r)
            
        return 0.0
    def _sample_transition(self, state, action):
        transitions = self.P[state][action]
        probs = [t[0] for t in transitions]
        i = random.choices(range(len(transitions)), weights=probs)[0]
        p, ns, r, terminated = transitions[i]
        return ns, r, terminated

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _get_next_state(self, state, action):
        transitions = self.P[state][action]  # [(prob, next_state, reward, terminated), ...]
        probs = [t[0] for t in transitions]
        idx = random.choices(range(len(transitions)), weights=probs)[0]
        p, next_state, reward, terminated = transitions[idx]

        return next_state, reward, terminated

    def _is_terminal(self, state):
        r = state // self.ncol
        c = state % self.ncol
        tile = self.desc[r, c]
        return tile in ("H", "G")


        
    def _get_best_action(self, root):
        # Exploitation: choose the child with the most visits (robustness)
        if not root.children:
            return random.choice([0, 1, 2, 3])
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        return sorted_children[0].action

# --- Main Game Loop ---
if __name__ == "__main__":
    # 1) Fast env for MCTS planning (NO rendering)
    env_plan = gym.make("FrozenLake-v1", is_slippery=True)


    # 2) Human env for visualization (render)
    env_vis = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")

    obs_plan, info = env_plan.reset(seed=0)
    obs_vis, info2 = env_vis.reset(seed=0)   # same seed so they start identical

    mcts = MCTS(env_plan)
    print(mcts.desc)

    done = False
    step_count = 0
    history = []

    print("Start MCTS Agent on *Slippery* Frozen Lake...")

    while not done:
        # PLAN (no render)
        print("obs_plan:", obs_plan, "terminal?", mcts._is_terminal(obs_plan))

        best_action, tree_root = mcts.search(initial_state=obs_plan, iterations=1000)

        # print("step", step_count, "state", observation, "is_terminal?", mcts._is_terminal(observation))


        # DEBUG: inspect what the tree actually contains
        print("root children:", len(tree_root.children))
        print("child visits:", [c.visits for c in tree_root.children])
        print("child types:", [getattr(c, "node_type", "NA") for c in tree_root.children])

        # (optional) record tree for LLM
        history.append({
            "step": step_count,
            "current_state": int(obs_plan),
            "chosen_action": int(best_action),
            "tree_snapshot": serialize_tree(tree_root, max_depth=3),
        })

        visualize_mcts_tree(tree_root,filename=f"frozen-lake/slippry_tree/png/mcts_step_{step_count}")

        # ACT in BOTH envs with the same action
        obs_plan, r_plan, term_plan, trunc_plan, info = env_plan.step(best_action)
        obs_vis, r_vis, term_vis, trunc_vis, info2 = env_vis.step(best_action)

        done = term_plan or trunc_plan
        step_count += 1

        # (optional) slow it down so you can watch
        import time; time.sleep(0.2)

        if done:
            print("Goal Reached!" if r_plan == 1 else "Fell in a hole.")

    with open("slippery_run_history.json", "w") as f:
        json.dump(history, f, indent=4)

    env_plan.close()
    env_vis.close()
