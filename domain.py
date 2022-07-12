import numpy as np
from math import pi, sqrt, atan2
import matplotlib.pyplot as plt

from teaming.Agent import Agent
from teaming.POI import POI
from scipy.interpolate import interp1d


class DiscreteRoverDomain:
    def __init__(self, N_agents, N_pois=4, poi_options=None, with_agents=True):
        if poi_options is None:
            poi_options = [[100, 1, 0]]
        self.N_agents = N_agents                    # number of agents
        self.n_agent_types = 1                      # TODO: update so it is not a dummy value
        self.N_pois = N_pois                        # number of POIs
        self.poi_options = np.array(poi_options)    # options for each POI - [refresh rate, number of observations required, ID]
        self.n_poi_types = np.shape(self.poi_options)[0]    # how many different types of POIs - allows for calc of L
        self.size = 10                              # size of the world
        self.time_steps = 100                       # time steps per epoch
        self.theoretical_max_g = 0
        self.pois = self.gen_pois()                 # generate POIs
        self.agents = self.gen_agents()             # generate agents
        self.avg_false = []
        self.with_agents = with_agents              # Include other agents in state / action space

        self.n_regions = 8                             # number of bins (quadrants) for sensor discretization
        self.sensor_bins = np.linspace(pi, -pi, self.n_regions + 1, True)  # Discretization for sensor bins
        self.sensor_range = 10
        self.reset()                                # reset the system
        self.vis = 0
        self.rand_action_rate = 0.05                # Percent of time a random action is chosen

    def draw(self):
        if self.vis == 0:
            self.vis = 1
            plt.ion()
            self.agent_offset = np.random.normal(0, 0.5, (self.N_agents, 2))
        plt.clf()
        xy = np.array([[poi.x, poi.y] for poi in self.pois])
        XY = np.array([[agent.x, agent.y] for agent in self.agents]) + self.agent_offset
        alpha = [1 - poi.refresh_idx / poi.refresh_rate for poi in self.pois]
        sizes = [poi.value * 20 + 10 for poi in self.pois]

        plt.scatter(xy[:, 0], xy[:, 1], marker="s", s=sizes, c=alpha, vmin=0, vmax=1)
        plt.scatter(XY[:, 0], XY[:, 1], marker="o")
        plt.ylim([0, self.size])
        plt.xlim([0, self.size])
        plt.pause(0.1)

    # generate list of agents
    def gen_agents(self):
        """
        Generates a list of agents
        :return: list of agent objects at random initial locations
        """
        # creates an array of x, y positions for each agent
        # locations are [0,4] plus half the size of the world
        X = np.random.randint(0, 2, self.N_agents) + self.size // 2
        Y = np.random.randint(0, 2, self.N_agents) + self.size // 2
        idxs = [i for i in range(self.N_agents)]
        # return an array of Agent objects at the specified locations
        # Agent initialization: x, y, idx, capabilties, type
        return [Agent(x, y, idx, np.random.random(self.N_pois), np.random.randint(0, self.n_agent_types))
                for x, y, idx in zip(X, Y, idxs)]

    # generate list of POIs
    def gen_pois(self):
        """
        Generates a list of random POIs
        :return: list of POI objects
        """
        num_poi_types = np.shape(self.poi_options)[0]
        randints = np.random.randint(num_poi_types,
                                     size=self.N_pois)  # get an array of random integers - integers represent the POI type, one for each POI in the world
        poi_vals = self.poi_options[randints, :]  # array of values from the POI options for each POI in the world
        x = np.random.randint(0, self.size, self.N_pois)  # x locations for all POIs
        y = np.random.randint(0, self.size, self.N_pois)  # y locations for all POIs
        # Each one is a different type
        n_agents = [self.N_agents] * self.N_pois

        # these need to be less hard-coded
        # refresh_rate = [10 for i in range(self.N_pois)]         # refresh rate for all POIs
        # obs_required = [2 for i in range(self.N_pois)]          # number of observations required
        # poi_type = [i for i in range(self.N_pois)]              # Each one is a different type
        # value = poi_type                                        # Value of the POI

        num_refreshes = np.ceil(self.time_steps / poi_vals[:, 0]) # how many times each POI can be captured
        self.theoretical_max_g = sum(num_refreshes)
        # print(self.theoretical_max_g)

        refresh_rate = np.ndarray.tolist(poi_vals[:, 0])
        obs_required = np.ndarray.tolist(poi_vals[:, 1])
        poi_type = np.ndarray.tolist(poi_vals[:, 2])

        couple = [1 for _ in range(self.N_pois)]  # coupling requirement for all POIs
        value = [1 for _ in range(self.N_pois)]  # Value of the POI
        poi_idx = [i for i in range(self.N_pois)]
        # return a list of the POI objects
        return list(map(POI, x, y, value, refresh_rate, obs_required, couple, poi_type, n_agents, poi_idx))

    # reset environment to intial config
    def reset(self):
        """
        Reset environment to initial configuration
        :return:
        """
        for a in self.agents:  # reset all agents to initial config
            a.reset()
        for p in self.pois:  # reset all POIs to initial config
            p.reset()
        self.avg_false = []

    def new_env(self):
        self.pois = self.gen_pois()
        self.agents = self.gen_agents()
        self.avg_false = []

    # perform one state transition given a list of actions for each agent
    def step(self, actions):
        """
        perform one state transition given a list of actions for each agent
        :param actions:
        :return:
        """
        # update all agents
        for i in range(self.N_agents):
            if actions[i]:
                self.agents[i].poi = actions[i]  # agents set a new goal at every time step
                self.agents[i].step()  # move agent toward POI

        # refresh all POIs and reset which agents are currently viewing
        for j in range(self.N_pois):
            self.pois[j].refresh()
            self.pois[j].viewing = []  # if this gets reset at every step, the "viewing" check will only see the last time step

    def run_sim(self, policies):
        """
        This is set up to run one epoch for the number of time steps specified in the class definition.
        Tests a set of NN policies, one for each agent.
        Parameters
        ----------
        policies: a policy (assumes a NN) for each agent

        Returns
        -------
        G: global reward
        """

        if len(policies) != self.N_agents:
            raise ValueError(
                'number of policies should equal number of agents in system (currently {})'.format(self.N_agents))

        for i in range(len(policies)):
            self.agents[i].policy = policies[i]

        for _ in range(self.time_steps):
            actions = []
            for agent in self.agents:
                self.state(agent)  # gets the state
                st = agent.state
                act_array = agent.policy(st).detach().numpy()  # picks an action based on the policy
                # act_detensorfied = np.argmax(
                #     act.detach().numpy())  # converts tensor to numpy, then finds the index of the max value
                act = self.action(agent, act_array)
                # TODO: add check to make sure action is valid
                actions.append(act)  # save the action to list of actions
            self.step(actions)
            self.avg_false.append(actions.count(False) / len(actions))

        # print("percent steps taken:", check / (self.time_steps * self.N_agents))
        return self.G(), np.mean(self.avg_false)

    def state(self, agent):
        """
        Takes in an agent, returns the state and the relevant indices for the closest POI or agent in each region of each type
        :param agent:
        :return state, state_idx:
        """

        # Number of sensor bins as rows, poi types plus agents types for columns
        n_agent_types = self.n_agent_types
        if not self.with_agents:
            n_agent_types = 0
        state = np.zeros((self.n_regions, len(self.poi_options) + n_agent_types)) - 1
        state_idx = np.zeros_like(state) - 1
        poi_dist, poi_quads = self._get_quadrant_state_info(agent, 'p')
        ag_dist, ag_quads = self._get_quadrant_state_info(agent, 'a')

        # Determine closest POI in each region
        for i in range(len(self.pois)):
            d = poi_dist[i]
            if d == -1:         # If the POI is out of range, skip it
                continue        # -1 was used as the arbitrary flag to indicate this POI is out of sensor range
            quad = poi_quads[i]
            poi_type = self.pois[i].poi_type
            # d is inverse distance, so this finds the closest one
            if d > state[quad, poi_type]:
                state[quad, poi_type] = d
                state_idx[quad, poi_type] = i

        # Determine closest agent in each region and sum inverse distances of all agents in each quadrant
        curr_best = np.zeros(self.n_regions) - 1
        if self.with_agents:
            for j in range(len(self.agents)):
                if self.agents[j] == agent:   # If looking at the current agent, skip it
                    continue

                d = ag_dist[j]
                if d == -1:         # If the agent is out of range, skip it
                    continue        # -1 was used as the arbitrary flag to indicate this agent is out of sensor range
                quad = ag_quads[j]
                ag_col = len(self.poi_options) + self.agents[j].type  # agent columns start after POI columns
                state[quad, ag_col] += d   # Sum of distances to all agents in that region
                if state[quad, ag_col] > 1:     # bound to [0, 1] - there is probably a more efficient check
                    state[quad, ag_col] = 1
                # Keeps track of the closest agent in each region
                if d > curr_best[quad]:
                    curr_best[quad] = d
                    state_idx[quad, ag_col] = j
        agent.state = state
        agent.state_idx = state_idx

    def _get_quadrant_state_info(self, agent, a_or_p='p'):
        """
        Get 'quadrant' state information for all other points or agents
        Parameters
        ----------
        agent:  agent for which we are getting state info
        a_or_p: is the state info for all other agents or POIs

        Returns
        -------
        distance and quadrant number for each point

        """
        # tested this and it should be accurate
        # Info for POIs
        if a_or_p == 'p':
            num_points = self.N_pois
            points = self.pois
        # Info for other agents
        else:
            num_points = self.N_agents - 1
            points = self.agents
        dist_arr = np.zeros(num_points)  # distance to each POI
        theta_arr = np.zeros(num_points)  # angle to each poi [-pi, pi]
        for i in range(num_points):
            # Compare each point (POI or other agent) to position of current agent
            # Check to make sure it is not the same agent happens later (ignore for now for indexing reasons)
            point = points[i]
            x = point.x - agent.x
            y = point.y - agent.y
            if x == 0 and y == 0:   # avoid divide by zero case
                inv_d = 2           # Set inverse distance to 2 - if <= 1 unit away, inv_d = 1. Need to differentiate "I am here"
            else:
                d = sqrt(x ** 2 + y ** 2)  # inverse of absolute distance to each POI
                inv_d = 1/d
                if inv_d > 1:   # limit to [0, 1] range
                    inv_d = 1
                if d > self.sensor_range:  # If it is out of sensor range, set it to -1 as a flag
                    inv_d = -1
            dist_arr[i] = inv_d
            theta_arr[i] = atan2(y, x)  # angle to each POI
        quadrants = np.digitize(theta_arr, bins=self.sensor_bins) - 1  # which quadrant / octant each POI is in relative to the GLOBAL frame
        return dist_arr, quadrants

    def state_size(self):
        """
        state size is the discretization of the sensor (number of bins) times the number of POI types plus one
        In each region, the sensor will have one bit for each POI type (distance) and one bit for the sum of the inverse distance to all other agents in that region
        :return:
        state size
        """
        if self.with_agents:
            return self.n_regions * (self.n_poi_types + self.n_agent_types)
        else:
            return self.n_regions * self.n_poi_types

    def action(self, agent, nn_output):
        """

        :param agent:
        :param nn_output: Assumes output from the NN a 1xN numpy array
        :return: agent, poi, or False (for no movement)
        """

        # Choose a random action a small percent of the time in order to get out of being stuck
        # If the agent policy has chosen not to move and the world isn't changing, the agent will never move.
        if np.random.random() < self.rand_action_rate:
            nn_max_idx = np.random.randint(0, len(nn_output))
        else:
            nn_max_idx = np.argmax(nn_output)

        # first (n_regions) number of outputs represent POIs
        if nn_max_idx < self.n_regions:
            # Get the index of the max value (aka min distance) POI in that region of any type
            state_idx = np.argmax(agent.state[nn_max_idx][:self.n_poi_types])
            if np.max(agent.state[nn_max_idx][state_idx]) == -1:
                # Flag that there's nothing of that type in that region
                return False
            # Find and return the closest POI in that region
            poi_idx = int(agent.state_idx[nn_max_idx][state_idx])
            return self.pois[poi_idx]

        # Second (n_regions) number of outputs represent agents
        elif nn_max_idx < self.n_regions * 2:
            if not self.with_agents:
                return False

            nn_max_idx -= self.n_regions
            # Need to get the region number (subtract n_reigons since this is the second set of them)
            # Get the index the max value (aka min distance) set of agents in that region of any type
            state_idx = np.argmax(agent.state[nn_max_idx][self.n_poi_types:]) + self.n_poi_types
            if np.max(agent.state[nn_max_idx][state_idx]) == -1:
                # Flag that there's nothing of that type in that region
                return False
            # Find and return the closest agent in that region
            # See not above - closest is much more vague when you're doing sum of inverse distances
            ag_idx = int(agent.state_idx[nn_max_idx][state_idx])
            return self.agents[ag_idx]

        # If the NN chose the dummy final option, then do nothing
        else:
            return False

    def get_action_size(self):
        """
        Output should be the number of regions in the sensor times two plus one.
        Output can choose the closest POI in each region, the closest agent in each region, or null (do nothing).
        :return:
        """
        if self.with_agents:
            return (self.n_regions * 2) + 1
        else:
            return self.n_regions + 1

    # returns global reward based on POI values
    def G(self):
        g = 0
        for poi in self.pois:
            g += poi.successes * poi.value
        return g

    def D(self):
        d = np.zeros(self.N_agents)
        for poi in self.pois:
            d = d + poi.D_vec * poi.value
        return d

    def Dpp(self):
        d = np.zeros(self.N_agents)
        for poi in self.pois:
            d = d + poi.Dpp_vec * poi.value
        return d


if __name__ == "__main__":
    np.random.seed(0)
    poi_types = [[100, 1, 0]]  # , [10, 3, 1], [50, 5, 2]]
    num_agents = 1
    num_pois = 1
    num_states = 2 ** (num_pois * 2)
    env = DiscreteRoverDomain(num_agents, num_pois, poi_types)

    for i in range(30):

        actions = [env.pois[0]]
        env.step(actions)

        # env.draw()
        print(i, env.G(), env.D())
    for agent in env.agents:
        print(env.state(agent))
