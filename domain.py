import numpy as np
from math import pi, sqrt, atan2
import matplotlib.pyplot as plt
from os import path, getcwd

from teaming.Agent import Agent
from teaming.POI import POI, FalsePOI
from time import sleep, time



class DiscreteRoverDomain:
    def __init__(self, p):
        self.p = p
        self.n_agents = p.n_agents                      # number of agents
        self.n_agent_types = p.n_agent_types            # number of types of agents
        self.n_pois = p.n_pois                          # number of POIs
        self.size = p.size                              # size of the world
        self.time_steps = p.time_steps                  # time steps per epoch
        self.n_rooms = p.n_rooms                        # number of rooms

        self.poi_options = np.array(p.poi_options)      # options for each POI - [refresh rate, number of observations required, ID]
        self.n_poi_types = np.shape(self.poi_options)[0]    # how many different types of POIs - allows for calc of L
        self.poi_x_y = []                               # to save POI xy locations
        self.theoretical_max_g = 0                      # Maximum G if all POIs were perfectly captured
        self.vis = 0                                    # To visualize or not to visualize
        self.time = 0
        self.pois = self.gen_pois()                     # generate POIs
        self.agents = self.gen_agents()                 # generate agents

        self.reset()                                # reset the system

    ################################# Setup Functions #################################
    def draw(self, t):
        if self.vis == 0:
            self.vis = 1
            plt.ion()
        agent_offset = np.random.normal(0, 0.5, (self.n_agents, 2))
        plt.clf()
        xy = np.array([[poi.x, poi.y] for poi in self.pois])
        XY = np.array([[agent.x, agent.y] for agent in self.agents]) + agent_offset
        # alpha = [1 - poi.refresh_idx / poi.refresh_rate for poi in self.pois]
        col = [poi.poi_type for poi in self.pois]
        alpha = np.array([poi.active for poi in self.pois], dtype=float)
        alpha[alpha < 0.1] = 0.2

        sizes = [poi.value * 20 + 10 for poi in self.pois]

        plt.scatter(xy[:, 0], xy[:, 1], marker="s", s=sizes, c=col, alpha=alpha, vmin=0, vmax=1)
        plt.scatter(XY[:, 0], XY[:, 1], marker="o")
        plt.ylim([0, self.size])
        plt.xlim([0, self.size])

        # plt.show()
        # sleep(0.1)
        pth = path.join(getcwd(), 'rollouts', self.p.fname_prepend + 't{:02d}_{}.png'.format(self.p.trial_num, t))

        plt.savefig(pth)

    def gen_agents(self):
        """
        Generates a list of agents
        :return: list of agent objects at random initial locations
        """
        # creates an array of x, y positions for each agent
        # locations are [0,4] plus half the size of the world
        X = np.random.randint(0, 2, self.n_agents) + self.size // 2
        Y = np.random.randint(0, 2, self.n_agents) + self.size // 2
        idxs = [i for i in range(self.n_agents)]
        # return an array of Agent objects at the specified locations
        # Agent initialization: x, y, idx, capabilties, type
        return [Agent(x, y, idx, np.random.random(self.n_pois), np.random.randint(0, self.n_agent_types), self.p)
                for x, y, idx in zip(X, Y, idxs)]

    def gen_pois(self):
        """
        Generates a list of random POIs
        :return: list of POI objects
        """
        num_poi_types = np.shape(self.poi_options)[0]
        self.captured = np.zeros(num_poi_types)

        #TODO: Will have to reconfigure this to put POIs in rooms / determine which POIs go in what rooms
        x = np.random.randint(0, self.size, self.n_pois)  # x locations for all POIs
        y = np.random.randint(0, self.size, self.n_pois)  # y locations for all POIs
        self.poi_x_y = np.array([x, y])     # Use this to save x,y locations
        params = [self.p for _ in range(self.p.n_pois)]
        # This block makes it so POIs are evenly distributed between types
        n_each_type = int(np.floor(self.n_pois / num_poi_types))    # number of each POI type
        poi_type = []
        for i in range(num_poi_types):
            poi_type += [i] * n_each_type
        while len(poi_type) < self.n_pois:
            poi_type += [np.random.randint(num_poi_types)]
        # Each one is a different type
        n_agents = [self.n_agents] * self.n_pois
        couple = [self.p.couple for _ in range(self.n_pois)]  # coupling requirement for all POIs
        obs_radius = [self.p.obs_radius for _ in range(self.n_pois)]  # Observation radius
        poi_idx = [j for j in range(self.n_pois)]
        # return a list of the POI objects
        return list(map(POI, x, y, couple, poi_type, n_agents, poi_idx, obs_radius, params))

    def save_poi_locs(self, fpath):
        np.save(fpath, self.poi_x_y)

    def move_pois(self):
        x = np.random.normal(0, 0.1, self.n_pois)  # x movement for all POIs
        y = np.random.normal(0, 0.1, self.n_pois)  # y movement for all POIs
        for i, poi in enumerate(self.pois):
            poi.x += x[i]
            poi.y += y[i]
            dims = [poi.x, poi.y]
            for d in dims:
                if d < 0:
                    d = 1
                elif d > self.size:
                    d = self.size - 1

    def reset(self):
        """
        Reset environment to initial configuration
        :return:
        """
        for a in self.agents:  # reset all agents to initial config
            a.reset()
        for p in self.pois:  # reset all POIs to initial config
            p.reset()

    def new_env(self):
        self.pois = self.gen_pois()
        self.agents = self.gen_agents()

    ################################# Run Sim Functions #################################
    def run_sim(self, policies):
        """
        This is set up to run one epoch for the number of time steps specified in the class definition.
        Tests a set of NN policies, one for each agent.
        Parameters
        ----------
        policies: a policy (assumes a NN) for each agent
        use_time: sets whether to use time in the agent state. Defaults to false

        Returns
        -------
        G: global reward
        """
        if len(policies) != self.n_agents:
            raise ValueError(
                f'number of policies should equal number of agents in system '
                f'(currently {self.n_agents} agents and {len(policies)} policies)')
        for i in range(len(policies)):
            self.agents[i].policy = policies[i]  # sets all agent policies
        for t in range(self.time_steps):
            self.time = t
            actions = self.joint_state()
            self.step(actions)
            if self.vis:
                self.draw(t)
        return self.G()

    def step(self, actions):
        """
        perform one state transition given a list of actions for each agent
        :param actions:
        :return:
        """
        # update all agents
        for i in range(self.n_agents):
            if actions[i]:
                self.agents[i].poi = actions[i]  # agents set a new goal at every time step
                self.agents[i].step()  # move agent toward POI

        # refresh all POIs and reset which agents are currently viewing
        for j in range(self.n_pois):
            self.pois[j].refresh()
            self.pois[
                j].viewing = []  # if this gets reset at every step, the "viewing" check will only see the last time step

    def state(self, agent):
        """
        Takes in an agent, returns the state and the relevant indices for the closest POI or agent in each region of each type
        :param agent:
        :return state, state_idx:
        """
        # Deleting this whole funciton because it will need to be rewritten from scratch
        pass

    def joint_state(self):
        actions = []
        for agent in self.agents:
            st = self.state(agent)  # calculates the state
            act_array = agent.policy(st).detach().numpy()  # picks an action based on the policy
            act = self.action(agent, act_array)
            actions.append(act)  # save the action to list of actions
        return actions

    def _get_quadrant_state_info(self, agent):
        """
        Get 'quadrant' state information for all other points or agents
        Parameters
        ----------
        agent:  agent for which we are getting state info
        Returns
        -------
        distance and quadrant number for each point

        """
        # Deleting code because it will need to be rewritten
        pass


    def state_size(self, use_time):
        """
        state size is the discretization of the sensor (number of bins) times the number of POI types plus one
        In each region, the sensor will have one bit for each POI type (distance) and one bit for the sum of the inverse distance to all other agents in that region
        :return:
        state size
        """
        return self.n_rooms * (self.n_poi_types + self.n_agent_types)


    def action(self, agent, nn_output):
        """

        :param agent:
        :param nn_output: Assumes output from the NN a 1xN numpy array
        :return: agent, poi, or False (for no movement)
        """
        nn_max_idx = np.argmax(nn_output)
        #TODO: Figure out the mapping between the NN output and the action
        pass


    def get_action_size(self):
        """
        Output should be the number of regions in the sensor times two plus one.
        Output can choose the closest POI in each region, the closest agent in each region, or null (do nothing).
        :return:
        """
        #TODO: Figure out the size of the NN output
        pass

    ################################# Reward Functions #################################
    def multiG(self):
        g = np.zeros(self.n_poi_types)
        for poi in self.pois:
            g[poi.poi_type] += poi.successes * poi.value
        return g

    # returns global reward based on POI values
    def G(self):
        g = 0
        for poi in self.pois:
            g += poi.successes * poi.value
        return g

    def sequence_G(self):
        g = 0
        if sum(self.captured) == self.n_poi_types:
            g = 1
        return g

    def D(self):
        d = np.zeros(self.n_agents)
        for poi in self.pois:
            d = d + poi.D_vec * poi.value
        return d

    def Dpp(self):
        d = np.zeros(self.n_agents)
        for poi in self.pois:
            d = d + poi.Dpp_vec * poi.value
        return d


if __name__ == "__main__":
    pass

