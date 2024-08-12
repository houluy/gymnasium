import torch
import itertools
import numpy as np
from numpy import linalg
from pettingzoo.mpe import simple_spread_v3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Agent:
    def __init__(self, position):
        self.position = position
        self.size = 0.15
        self.color = "blue"


class SimpleSpread:
    def __init__(self, env, no_comm=True):
        self.env = env
        self.env.reset() # Only for initialization
        self.agent_list = env.agents[:]
        self.agent_size = 0.15  # This is the size of the agent, initialized in Scenario
        self.landmark_num = self.agent_num = env.num_agents
        self.agent_position_ind = 2
        self.landmark_rel_position_ind = self.agent_position_ind + 2
        self.landmark_size = self.landmark_threshold_dist = 0.1
        self.other_agent_rel_position_ind = self.landmark_rel_position_ind + 2 * self.landmark_num
        self.communication_feature_num = 4
        self.communication_feature_ind = self.other_agent_rel_position_ind + 2 * (self.agent_num - 1)
        self.global_observation_simple_feature_num = self.agent_num * 4 + self.landmark_num * 2  # Agent position, velocity, landmark position

        if no_comm:
            self.local_observation_feature_num = env.observation_space("agent_0").shape[0] - self.communication_feature_num
            self.global_observation_feature_num = self.local_observation_feature_num * self.agent_num
        else:
            self.local_observation_feature_num = env.observation_space("agent_0").shape[0]
            self.global_observation_feature_num = self.local_observation_feature_num * self.agent_num
        # Discrete action space
        self.local_action_num = env.action_space("agent_0").n
        self.global_action_num = self.local_action_num * self.agent_num
        self.action_map = list(itertools.combinations_with_replacement(range(self.local_action_num), self.agent_num))

        # Continuous action space
        # TBD

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    @property
    def agents(self):
        return self.env.agents

    def _remove_comm_feature(self, observation):
        return observation[:-self.communication_feature_num]

    def local2global_observation(self, observations: dict):
        # Notice: Directly concatenate the observations of all agents
        return torch.cat([torch.from_numpy(self._remove_comm_feature(observation)) for agent, observation in observations.items()])

    def local2global_observation_simple(self, observations: dict):
        # Notice: Agent positions, velocities, landmark positions
        global_observation = np.empty((0,))
        counter = 0
        landmark_positions = None
        agent_positions = np.empty((0,))
        agent_velocities = np.empty((0,))
        for agent, observation in observations.items():
            velocity, position, landmark_rel_positions, other_agent_rel_positions = self.decomposite_local_observation(observation)
            agent_positions = np.concatenate((agent_positions, position))
            agent_velocities = np.concatenate((agent_velocities, velocity))
            if counter == 0:
                landmark_positions = np.tile(position, self.landmark_num) + landmark_rel_positions
            counter += 1
            global_observation = np.concatenate((global_observation, velocity, position))
        global_observation = np.concatenate((global_observation, landmark_positions))
        return global_observation.astype(np.float32).reshape((self.global_observation_simple_feature_num, )), agent_positions.reshape((self.landmark_num, 2)), landmark_positions.reshape((self.landmark_num, 2)), agent_velocities.reshape((self.agent_num, 2))

    def discrete_action_assignment(self, global_action: int):
        action_list = self.action_map[global_action]
        return {
            agent: action_list[agent_ind] for agent_ind, agent in enumerate(self.agent_list)
        }

    def decomposite_local_observation(self, observation):
        velocity = observation[:self.agent_position_ind]
        position = observation[self.agent_position_ind:self.landmark_rel_position_ind]
        landmark_rel_positions = observation[self.landmark_rel_position_ind:self.other_agent_rel_position_ind]
        other_agent_rel_positions = observation[self.other_agent_rel_position_ind:self.communication_feature_ind]
        return velocity, position, landmark_rel_positions, other_agent_rel_positions

    def benchmark_data(self, observations: dict):
        """
        :param observations: dict of agent_id: observation
        %return: 
            avg_d
            occupied_landmarks
            collisions
        """
        avg_d = 0
        occupied_landmarks = 0
        collisions = 0
        for agent, observation in observations.items():
            velocity, position, landmark_rel_positions, other_agent_rel_positions = self.decomposite_local_observation(observation)
            all_dists_landmark = []
            all_dists_collision = []
            for landmark_ind in range(self.landmark_num):
                all_dists_landmark.append(linalg.norm(landmark_rel_positions[landmark_ind * 2: landmark_ind * 2 + 2]))
            min_dist = min(all_dists_landmark)
            if min_dist < self.landmark_threshold_dist:
                occupied_landmarks += 1
            avg_d += min_dist
            for other_agent_ind in range(self.agent_num - 1):
                dis = linalg.norm(other_agent_rel_positions[other_agent_ind * 2: other_agent_ind * 2 + 2])
                if dis < 2 * self.agent_size:
                    collisions += 1

        return avg_d / self.agent_num, occupied_landmarks, collisions // 2

    def show(self, agent_positions, landmark_positions, agent_velocities):
        figure, axes = plt.subplots()
        for ind, agent_position in enumerate(agent_positions):
            c = mpatches.Circle(agent_position, radius=self.agent_size, color='blue', fill=False, label="agent")
            plt.gcf().gca().add_artist(c)
            plt.arrow(agent_position[0], agent_position[1], agent_velocities[ind][0], agent_velocities[ind][1], head_width=0.05, head_length=0.1, fc='blue', ec='blue')

        for landmark_position in landmark_positions:
            c = mpatches.Circle(landmark_position, radius=self.landmark_size, color='red', fill=True, label="landmark")
            plt.gca().add_artist(c)
        #plt.legend()
        plt.title("Show agnets and landmarks")
        axes.set_xlim(-2, 2)
        axes.set_ylim(-2, 2)
        axes.set_aspect(1)
        plt.show()


if __name__ == "__main__":
    env = simple_spread_v3.parallel_env(render_mode=None)
    ss = SimpleSpread(env)

    observations, infos = env.reset()

    benchmark_data = ss.benchmark_data(observations)
    global_observation = ss.local2global_observation(observations)
    global_observation_simple, agent_positions, landmark_positions, agent_velocities = ss.local2global_observation_simple(observations)
    ss.show(agent_positions, landmark_positions, agent_velocities)
