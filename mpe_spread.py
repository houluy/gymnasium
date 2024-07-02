from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    print("----------------------")
    print(agent)
    observation, reward, termination, truncation, info = env.last()
    print(observation)
    print(reward)

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()
        print(action)

    env.step(action)
env.close()