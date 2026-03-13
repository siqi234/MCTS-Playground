import json
import graphviz
import os

# 全局计数器，用于给每个图节点生成唯一的 ID
node_counter = 0

def json_to_png(json_filepath, output_filename, min_visits=5):
    """
    读取 MCTS 生成的 JSON 文件，并渲染成包含 wins 和 holes 数据的 PNG 树状图
    """
    if not os.path.exists(json_filepath):
        print(f"找不到文件: {json_filepath}")
        return

    with open(json_filepath, 'r', encoding='utf-8') as f:
        tree_data = json.load(f)

    # 初始化 Graphviz 有向图
    dot = graphviz.Digraph(comment='MCTS Search Tree')
    dot.attr(rankdir='TB', dpi='300') # TB: Top to Bottom, 高分辨率

    # 动作映射字典
    action_names = {"0": "Left", "1": "Down", "2": "Right", "3": "Up"}

    def traverse(node_data, parent_id=None, edge_label=""):
        global node_counter
        
        # 1. 过滤：如果是“达到最大深度”的字符串，或者访问次数太低，直接跳过不画
        if isinstance(node_data, str) or node_data.get("visits", 0) < min_visits:
            return

        # 2. 生成当前节点的唯一 ID
        current_id = str(node_counter)
        node_counter += 1

        node_type = node_data.get("type")
        visits = node_data.get("visits", 0)
        value = node_data.get("value", 0.0)
        successes = node_data.get("success", 0)
        holes = node_data.get("failure", 0)

        # 3. 绘制红点 (DecisionNode) 或绿点 (ChanceNode)
        if node_type == "DecisionNode":
            state = node_data.get("state")
            # 【核心展示】：直接把 wins 和 holes 贴在节点上
            label = f"State: {state}\nVisits: {visits}\nValue: {value:.2f}\nWins: {successes} | Holes: {holes}"
            dot.node(current_id, label, shape='ellipse', style='filled', fillcolor='lightblue')
            
            # 连线到父节点
            if parent_id is not None:
                dot.edge(parent_id, current_id, label=edge_label)
                
            # 遍历子节点 (ChanceNodes)
            children = node_data.get("children", {})
            if isinstance(children, dict):
                for act_id_str, child_data in children.items():
                    act_name = action_names.get(act_id_str, f"Act {act_id_str}")
                    traverse(child_data, current_id, act_name)
                    
        elif node_type == "ChanceNode":
            act_id_str = str(node_data.get("action_id"))
            act_name = action_names.get(act_id_str, f"Act {act_id_str}")
            label = f"Action: {act_name}\nVisits: {visits}\nValue: {value:.2f}\nWins: {successes} | Holes: {holes}"
            dot.node(current_id, label, shape='box', style='filled', fillcolor='lightgreen')
            
            # 连线到父节点
            if parent_id is not None:
                dot.edge(parent_id, current_id, label=edge_label)
                
            # 遍历子节点 (DecisionNodes)
            children = node_data.get("children", {})
            if isinstance(children, dict):
                for next_state_str, child_data in children.items():
                    traverse(child_data, current_id, f"to {next_state_str}")

    print(f"正在解析 {json_filepath} 并生成树状图...")
    # 从根节点开始递归
    traverse(tree_data)

    # 渲染并输出 PNG (cleanup=True 会删掉生成的中间 .dot 文本文件)
    dot.render(output_filename, format='png', cleanup=True)
    print(f"✅ 成功生成图片：{output_filename}.png")


if __name__ == "__main__":
    # 假设你的主程序跑完后，在 mcts_trees 文件夹下生成了 step_0 的 json
    input_json = "mcts_trees_ver2/mcts_tree_step_0.json"
    output_png = "mcts_trees_ver2/visualized_step_0"
    
    # min_visits 阈值：如果图太拥挤，可以把 5 调大（比如 20）；如果图太空，调小（比如 1）
    json_to_png(input_json, output_png, min_visits=5)