import gymnasium as gym

# env_name = "CliffWalking-v0"

env_name = "MountainCarContinuous-v0"

env = gym.make(env_name, render_mode="human")

done = False
observation = env.reset()
env.render()
while not done:
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    env.render()
