from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.env(render_mode="human")
env.reset(seed=42)
unwrapped = env.unwrapped
# State Example (Per agent)

# [ 0.46382314 -0.82866174 -0.23761715 -0.7907784   0.7598966   1.362907
# -0.50615555  0.6915503  -0.0207868   1.6443084   1.5769814   1.9791899
# -0.17950885  1.0424795   0.          0.          0.          0.        ]

# 1. self.vel: [0.46382314 -0.82866174]
# 2. self.pos: [-0.23761715 -0.7907784 ]
# 3. landmark_rel_position: [ 0.7598966   1.362907   -1.50615555  0.6915503  -0.0207868   1.6443084]
# 3. other_agent_rel_positions: [1.5769814  1.9791899, -0.17950885, 1.0424795]
# 5. communications: [0. 0. 0. 0.]

# Action Example (Per agent)
# Discrete
# [0, 1, 2, 3, 4]

# Single agent
# Global state -> all actions

# Independent agents (CTDE)
# Local state -> local action

# Communication agents
# Local state + message -> local action + message

def aec():
    i = 0
    for agent in env.agent_iter():
        breakpoint()
        print("----------------------")
        print(agent)
        observation, reward, termination, truncation, info = env.last()
        print(observation)
        print(reward)
        print(termination, truncation)
        breakpoint()
        i += 1
        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()
            print(action)

        res = env.step(action)
        print(res)
        breakpoint()
    env.close()
    print(i)


def parallel():
    env = simple_spread_v3.parallel_env(render_mode="human")
    observations, infos = env.reset()
    breakpoint()

    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()


#breakpoint()
#parallel()
#aec()