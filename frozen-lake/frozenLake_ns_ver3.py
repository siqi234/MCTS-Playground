import os
import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import gymnasium as gym
from graphviz import Digraph

ACTION_MAP = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}


# ----------------------------
# True chance-node tree classes
# ----------------------------

@dataclass
class OutcomeEdge:
    """One stochastic outcome from a ChanceNode: prob -> next DecisionNode."""
    prob: float
    next_state: int
    reward: float
    done: bool


@dataclass
class ChanceNode:
    """
    ChanceNode corresponds to a specific (state s, action a).
    It contains ALL possible outcomes from env.unwrapped.P[s][a].
    """
    s: int
    a: int
    outcomes: List[OutcomeEdge]

    visits: int = 0
    value: float = 0.0  # total return passing through this chance node

    # next_state -> DecisionNode
    children: Dict[int, "DecisionNode"] = field(default_factory=dict)

    # bookkeeping: how many times each outcome was sampled
    outcome_visits: Dict[int, int] = field(default_factory=dict)

    def sample_outcome(self) -> OutcomeEdge:
        """Sample an outcome according to its probability."""
        r = random.random()
        cum = 0.0
        for oe in self.outcomes:
            cum += oe.prob
            if r <= cum:
                return oe
        return self.outcomes[-1]  # float safety

    def get_child(self, next_state: int) -> "DecisionNode":
        if next_state not in self.children:
            self.children[next_state] = DecisionNode(state=next_state, parent_chance=self)
        return self.children[next_state]


@dataclass
class DecisionNode:
    """
    DecisionNode corresponds to an environment state s.
    Children are ChanceNodes, one per action.
    """
    state: int
    parent_chance: Optional[ChanceNode] = None  # how we got here (chance edge)

    visits: int = 0
    value: float = 0.0  # total return passing through this state

    # action -> ChanceNode
    chance_children: Dict[int, ChanceNode] = field(default_factory=dict)

    # action stats for UCT: N(s,a) and Q(s,a)=mean return
    N_sa: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    Q_sa: Dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})

    # track which actions have not been expanded into ChanceNode yet
    untried_actions: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0


# ----------------------------
# MCTS with explicit chance nodes
# ----------------------------

class TrueChanceMCTS:
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        c_uct: float = 1.4,
        max_rollout_steps: int = 50,
    ):
        self.env = env
        self.P = env.unwrapped.P  # P[s][a] = [(prob, s2, reward, done), ...]
        self.gamma = gamma
        self.c = c_uct
        self.max_rollout_steps = max_rollout_steps

        # terminal states from map
        desc = env.unwrapped.desc
        n = desc.shape[0]
        self.holes = set()
        self.goal = None
        for r in range(n):
            for c in range(n):
                ch = desc[r, c].decode("utf-8")
                s = r * n + c
                if ch == "H":
                    self.holes.add(s)
                elif ch == "G":
                    self.goal = s

    def is_terminal_state(self, s: int) -> bool:
        return (s in self.holes) or (self.goal is not None and s == self.goal)

    def ensure_chance_node(self, dn: DecisionNode, a: int) -> ChanceNode:
        if a in dn.chance_children:
            return dn.chance_children[a]

        raw = self.P[dn.state][a]
        outcomes = [OutcomeEdge(prob=p, next_state=s2, reward=r, done=done) for (p, s2, r, done) in raw]
        cn = ChanceNode(s=dn.state, a=a, outcomes=outcomes)
        dn.chance_children[a] = cn
        return cn

    def uct_action(self, dn: DecisionNode) -> int:
        """
        Select action at a decision node using UCT over action stats Q_sa / N_sa.
        """
        best_a = None
        best_score = -1e18

        for a in [0, 1, 2, 3]:
            n_sa = dn.N_sa[a]
            if n_sa == 0:
                score = float("inf")
            else:
                exploit = dn.Q_sa[a]
                explore = self.c * math.sqrt(math.log(max(1, dn.visits)) / n_sa)
                score = exploit + explore

            if score > best_score:
                best_score = score
                best_a = a

        return int(best_a)

    def rollout(self, start_state: int) -> float:
        """
        Rollout using the true transition model P (includes chance).
        Reward is taken from transitions; terminal uses done in transitions.
        """
        G = 0.0
        discount = 1.0
        s = start_state

        for _ in range(self.max_rollout_steps):
            if self.is_terminal_state(s):
                break

            a = random.choice([0, 1, 2, 3])
            trans = self.P[s][a]  # list of (p,s2,r,done)

            # sample an outcome
            r = random.random()
            cum = 0.0
            chosen = trans[-1]
            for t in trans:
                cum += t[0]
                if r <= cum:
                    chosen = t
                    break
            p, s2, rew, done = chosen

            G += discount * rew
            discount *= self.gamma
            s = s2
            if done:
                break

        return G

    def simulate(self, root: DecisionNode, tree_depth_limit: int = 50) -> Dict[str, Any]:
        """
        One MCTS simulation that alternates:
          DecisionNode --(choose action)--> ChanceNode --(sample outcome)--> DecisionNode
        Returns a record dict (for JSON logging).
        """
        path_decisions: List[DecisionNode] = []
        path_actions: List[int] = []
        path_chance: List[Tuple[ChanceNode, int]] = []  # (chance_node, sampled_next_state)
        path_rewards: List[float] = []  # immediate reward on sampled outcome

        dn = root

        # Selection + Expansion
        leaf_return = None
        for _ in range(tree_depth_limit):
            if self.is_terminal_state(dn.state):
                leaf_return = 0.0
                break

            path_decisions.append(dn)

            # Expand a new action if possible
            if not dn.is_fully_expanded():
                a = dn.untried_actions.pop()  # expand this action into a chance node
            else:
                a = self.uct_action(dn)

            path_actions.append(a)

            cn = self.ensure_chance_node(dn, a)

            # Sample chance outcome (this is the TRUE chance node)
            oe = cn.sample_outcome()
            cn.visits += 1
            cn.outcome_visits[oe.next_state] = cn.outcome_visits.get(oe.next_state, 0) + 1

            path_chance.append((cn, oe.next_state))
            path_rewards.append(oe.reward)

            # Move to next decision node
            dn = cn.get_child(oe.next_state)

            # Expansion stop rule: first time we see this new decision state node, stop and rollout
            if dn.visits == 0:
                # leaf_return includes immediate reward + rollout from next state
                leaf_return = oe.reward + self.gamma * (0.0 if oe.done else self.rollout(dn.state))
                break

            if oe.done:
                leaf_return = oe.reward
                break

        if leaf_return is None:
            leaf_return = self.rollout(dn.state)

        # Backprop
        G = leaf_return
        for i in reversed(range(len(path_decisions))):
            dn_i = path_decisions[i]
            a_i = path_actions[i]
            cn_i, next_s_i = path_chance[i]
            rew_i = path_rewards[i]

            # update decision node
            dn_i.visits += 1
            dn_i.value += G

            # update action stats at decision node (incremental mean)
            dn_i.N_sa[a_i] += 1
            dn_i.Q_sa[a_i] += (G - dn_i.Q_sa[a_i]) / dn_i.N_sa[a_i]

            # update chance node stats
            cn_i.value += G  # total return through chance node

            # move one step back in return if you want step-wise discounting;
            # here we keep simplest "same G" style like many toy MCTS examples.

        # build simulation record for JSON
        sim_record = {
            "path": [
                {
                    "decision_state": int(path_decisions[i].state),
                    "action": int(path_actions[i]),
                    "action_name": ACTION_MAP.get(path_actions[i], "?"),
                    "chance_state_action": f"({path_decisions[i].state},{path_actions[i]})",
                    "sampled_next_state": int(path_chance[i][1]),
                    "immediate_reward": float(path_rewards[i]),
                }
                for i in range(len(path_decisions))
            ],
            "return_G": float(G),
        }
        return sim_record

    def search(self, initial_state: int, iterations: int = 1000) -> Tuple[int, DecisionNode]:
        root = DecisionNode(state=int(initial_state))

        for _ in range(iterations):
            self.simulate(root)

        # pick most visited action (robust child)
        best_a = max([0, 1, 2, 3], key=lambda a: root.N_sa[a])
        return int(best_a), root


# ----------------------------
# Serialization & Visualization (includes chance nodes)
# ----------------------------

def serialize_tree_with_chance(node: DecisionNode, max_depth: int = 3) -> Optional[Dict[str, Any]]:
    if node is None or max_depth < 0:
        return None

    win_rate = (node.value / node.visits) if node.visits > 0 else 0.0

    # Decision node
    out = {
        "type": "Decision",
        "state": int(node.state),
        "visits": int(node.visits),
        "total_value": float(node.value),
        "win_rate": round(float(win_rate), 4),
        "actions": [],
    }

    for a in [0, 1, 2, 3]:
        entry = {
            "action": int(a),
            "action_name": ACTION_MAP.get(a, "?"),
            "N_sa": int(node.N_sa[a]),
            "Q_sa": float(node.Q_sa[a]),
            "chance": None,
        }

        cn = node.chance_children.get(a)
        if cn is not None:
            cn_wr = (cn.value / cn.visits) if cn.visits > 0 else 0.0
            chance_dict = {
                "type": "Chance",
                "state_action": f"({cn.s},{cn.a})",
                "visits": int(cn.visits),
                "total_value": float(cn.value),
                "win_rate": round(float(cn_wr), 4),
                "outcomes": [],
            }

            # outcomes in the true model
            for oe in cn.outcomes:
                child_dn = cn.children.get(oe.next_state)
                sampled_n = cn.outcome_visits.get(oe.next_state, 0)
                outcome_item = {
                    "prob": float(oe.prob),
                    "next_state": int(oe.next_state),
                    "reward": float(oe.reward),
                    "done": bool(oe.done),
                    "sampled_visits": int(sampled_n),
                    "child": serialize_tree_with_chance(child_dn, max_depth - 1) if child_dn else None,
                }
                chance_dict["outcomes"].append(outcome_item)

            entry["chance"] = chance_dict

        out["actions"].append(entry)

    return out


def visualize_tree_with_chance(root: DecisionNode, filename: str, min_visits_to_plot: int = 5, max_nodes: int = 400):
    dot = Digraph(comment="True Chance-Node MCTS Tree")
    dot.attr(rankdir="LR")

    # decision nodes as boxes, chance nodes as ellipses
    drawn = 0
    queue: List[Tuple[str, str, Any]] = [("D", "Root", root)]
    dot.node("Root", label=f"D State:{root.state}\nN:{root.visits}", shape="box")
    drawn += 1

    idx = 0
    while queue and drawn < max_nodes:
        typ, node_id, obj = queue.pop(0)

        if typ == "D":
            dn: DecisionNode = obj
            for a in [0, 1, 2, 3]:
                if dn.N_sa[a] < min_visits_to_plot:
                    continue
                cn = dn.chance_children.get(a)
                if cn is None:
                    continue
                idx += 1
                cid = f"C{idx}"
                dot.node(cid, label=f"C ({dn.state},{a})\nN:{cn.visits}", shape="ellipse")
                dot.edge(node_id, cid, label=f"a={a} {ACTION_MAP[a]}\nNsa={dn.N_sa[a]}\nQ={dn.Q_sa[a]:.3f}")
                drawn += 1

                queue.append(("C", cid, cn))
                if drawn >= max_nodes:
                    break

        else:
            cn: ChanceNode = obj
            # show all outcomes (or a subset if you want)
            for oe in cn.outcomes:
                child = cn.children.get(oe.next_state)
                if child is None:
                    continue
                if child.visits < min_visits_to_plot:
                    continue
                idx += 1
                did = f"D{idx}"
                wr = (child.value / child.visits) if child.visits else 0.0
                dot.node(did, label=f"D State:{child.state}\nN:{child.visits}\nWR:{wr:.3f}", shape="box")
                dot.edge(node_id, did, label=f"p={oe.prob:.3f}\nr={oe.reward}\ndone={oe.done}")
                drawn += 1

                queue.append(("D", did, child))
                if drawn >= max_nodes:
                    break

    dot.render(filename, format="png", cleanup=True)
    print(f"Saved: {filename}.png  (nodes={drawn}, min_visits={min_visits_to_plot})")


# ----------------------------
# Demo: run episode, save per-step PNG + JSON
# ----------------------------

if __name__ == "__main__":
    out_dir = "frozen-lake/chance_tree"
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human")
    obs, info = env.reset()

    mcts = TrueChanceMCTS(env, gamma=0.99, c_uct=1.4, max_rollout_steps=50)

    history = []
    done = False
    step = 0

    print("Start TRUE chance-node MCTS on slippery FrozenLake...")

    while not done and step < 100:
        action, root = mcts.search(initial_state=int(obs), iterations=1500)

        # snapshot to JSON (includes decision+chance layers)
        snap = serialize_tree_with_chance(root, max_depth=2)

        step_data = {
            "step": step,
            "current_state": int(obs),
            "chosen_action": int(action),
            "chosen_action_name": ACTION_MAP[action],
            "tree_snapshot": snap,
        }
        history.append(step_data)

        # PNG for this step
        visualize_tree_with_chance(
            root,
            filename=os.path.join(out_dir, f"mcts_step_{step:03d}"),
            min_visits_to_plot=5,
            max_nodes=350,
        )

        # act in the real env
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            print("Goal Reached!" if reward == 1 else "Fell in a hole.")

        step += 1

    # write history JSON
    json_path = os.path.join(out_dir, "slippery_run_history_chance.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print("Saved JSON:", json_path)
    env.close()