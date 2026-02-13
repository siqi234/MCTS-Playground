import gymnasium as gym

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="ansi")
env.reset()
print(env.render())
