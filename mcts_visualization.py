import json
import graphviz
import os

# Global counter for unique node IDs in the Graphviz visualization
node_counter = 0

def json_to_png(json_filepath, output_filename="mcts_tree", min_visits=5):
    """
    Convert a JSON file representing an MCTS tree into a PNG image using Graphviz.
    """
    if not os.path.exists(json_filepath):
        print(f"Not found: {json_filepath}")
        return

    with open(json_filepath, 'r', encoding='utf-8') as f:
        tree_data = json.load(f)

    # Initialize a Graphviz Digraph
    dot = graphviz.Digraph(comment='MCTS Search Tree')
    dot.attr(rankdir='TB', dpi='300') # Set top-to-bottom layout and high resolution

    # Define action names for better visualization
    action_names = {"0": "Left", "1": "Down", "2": "Right", "3": "Up"}

    def add_nodes_edges(node_data, parent_id=None, edge_label=""):
        global node_counter
        
        # 1. Filter: If it's a string indicating "reached maximum depth" or visit count is too low, skip
        if isinstance(node_data, str) or node_data.get("visits", 0) < min_visits:
            return

        # 2. Generate a unique ID for the current node
        current_id = str(node_counter)
        node_counter += 1

        node_type = node_data.get("type")
        visits = node_data.get("visits", 0)
        value = node_data.get("value", 0.0)

        # 3. Draw the node
        if node_type == "DecisionNode":
            state = node_data.get("state")
            label = f"State: {state}\nVisits: {visits}\nValue: {value:.2f}"
            dot.node(current_id, label, shape='ellipse', style='filled', fillcolor='lightblue')
            
        elif node_type == "ChanceNode":
            action_id = str(node_data.get("action_id"))
            act_str = action_names.get(action_id, f"Act {action_id}")
            label = f"Action: {act_str}\nVisits: {visits}\nValue: {value:.2f}"
            dot.node(current_id, label, shape='box', style='filled', fillcolor='lightgreen')

        # 4. Draw the edge from the parent node to the current node with the appropriate label
        if parent_id is not None:
            dot.edge(parent_id, current_id, label=edge_label)

        # 5. Recursively traverse child nodes
        children = node_data.get("children", {})
        if isinstance(children, dict):
            for key, child_data in children.items():
                if node_type == "DecisionNode":
                    # Decision node's children are Chance nodes, edge label is the action taken
                    edge_text = action_names.get(key, f"Act {key}")
                else:
                    # Chance node's children are Decision nodes, edge label represents the state transition
                    edge_text = f"to {key}"
                    
                add_nodes_edges(child_data, current_id, edge_text)

    print("Generating tree visualization...")
    add_nodes_edges(tree_data)

    # output png
    dot.render(output_filename, format='png', cleanup=True)
    print(f"Successfully generated image: {output_filename}.png")

if __name__ == "__main__":


    for file in os.listdir("mcts_trees"):
        if file.endswith(".json"):
            print(f"Found JSON file: {file}")
            input_json = os.path.join("mcts_trees", file)
            output_name = f"mcts_trees/png/{os.path.splitext(file)[0]}"
            json_to_png(input_json, output_filename=output_name, min_visits=5)