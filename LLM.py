import os
import openai
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), 
                base_url="https://api.groq.com/openai/v1")
json_file = 'mcts_trees_ver2/mcts_tree_step_0.json'

with open(json_file, 'r') as f:
    json_data = json.load(f)

prompt = f"Here is the MCTS tree for step one: {json_data}, using sentences within 200 word that a non-technical person can understand, explain why the agent choose their first action in the Frozen lake game."

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile", 
    messages=[
        {"role": "system", "content": "you are a decision system explainer, "
        "you will be given a JSON data that represents a decision tree for one time step of a Frozen Lake game(slippery), "
        "and you need to explain the decision process in the tree when the user asks."},
        {"role": "user", "content": prompt}
    ]
)

explanation = response.choices[0].message.content
print("System Explanation:", explanation)