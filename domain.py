import numpy as np
from math import pi, sqrt, atan2
import matplotlib.pyplot as plt
from os import path, getcwd

from teaming.Agent import Agent
from teaming.POI import POI, FalsePOI
from time import sleep, time



class DiscreteRoverDomain:
    def __init__(self, p):
        if p.poi_options is None:
            p.poi_options = [[100, 1, 0]]
        self.p = p
        self.n_agents = p.n_agents                      # number of agents
        self.n_agent_types = p.n_agent_types            # number of types of agents
        self.n_pois = p.n_pois                          # number of POIs
        self.size = p.size                              # size of the world
        self.time_steps = p.time_steps                  # time steps per epoch
        self.with_agents = p.with_agents                # Include other agents in state / action space
        self.n_regions = p.n_regions                    # number of bins (quadrants) for sensor discretization
        self.sensor_range = p.sensor_range
        self.rand_action_rate = p.rand_action_rate      # Percent of time a random action is chosen

        self.poi_x_y = []                               # to save POI xy locations
        self.avg_false = []                             # How many times agents choose null actions
        self.theoretical_max_g = 0                      # Maximum G if all POIs were perfectly captured
        self.vis = 0                                    # To visualize or not to visualize
        self.visualize = False
        self.poi_options = np.array(p.poi_options)      # options for each POI - [refresh rate, number of observations required, ID]
        self.n_poi_types = np.shape(self.poi_options)[0]    # how many different types of POIs - allows for calc of L
        self.pois = self.gen_pois()                     # generate POIs
        self.agents = self.gen_agents()                 # generate agents
        self.sensor_bins = np.linspace(pi, -pi, self.n_regions + 1, True)  # Discretization for sensor bins

        self.reset()                                # reset the system

    def draw(self, t):
        if self.vis == 0:
            self.vis = 1
            plt.ion()
            self.agent_offset = np.random.normal(0, 0.5, (self.n_agents, 2))
        plt.clf()
        xy = np.array([[poi.x, poi.y] for poi in self.pois])
        XY = np.array([[agent.x, agent.y] for agent in self.agents]) + self.agent_offset
        # alpha = [1 - poi.refresh_idx / poi.refresh_rate for poi in self.pois]
        alpha = [poi.poi_type for poi in self.pois]
        sizes = [poi.value * 20 + 10 for poi in self.pois]

        plt.scatter(xy[:, 0], xy[:, 1], marker="s", s=sizes, c=alpha, vmin=0, vmax=1)
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
        n_each_type = int(np.floor(self.n_pois / num_poi_types))

        # This block makes it so POIs are evenly distributed between types
        poi_type = []
        for i in range(num_poi_types):
            poi_type += [i] * n_each_type
        while len(poi_type) != self.n_pois:
            poi_type += [np.random.randint(num_poi_types)]
        # Instead of randomly assigned like the original line below:
        # poi_type = np.random.randint(num_poi_types,
        #                              size=self.n_pois)  # integers represent the POI type, one for each POI in the world
        poi_vals = self.poi_options[poi_type, :]  # array of values from the POI options for each POI in the world
        x = np.random.randint(0, self.size, self.n_pois)  # x locations for all POIs
        y = np.random.randint(0, self.size, self.n_pois)  # y locations for all POIs
        self.poi_x_y = np.array([x, y])

        # x = [2, 8]
        # y = [2, 8]
        # Each one is a different type
        n_agents = [self.n_agents] * self.n_pois

        time_active = np.ndarray.tolist(poi_vals[:, 0])
        n_times_active = np.array(poi_vals[:, 1])
        value = np.ndarray.tolist(poi_vals[:, 3])  # Use this if you want to set the values of each POI type individually
        self.theoretical_max_g = sum(n_times_active * value)
        slot_size = np.ndarray.tolist(np.floor(self.time_steps / n_times_active))      # should calculate equal length time slots
        obs_required = np.ndarray.tolist(poi_vals[:, 2])
        # value = [self.p.value for _ in range(self.n_pois)]  # Use this if you want all values to be the same

        couple = [self.p.couple for _ in range(self.n_pois)]  # coupling requirement for all POIs
        obs_radius = [self.p.obs_radius for _ in range(self.n_pois)]  # Observation radius
        poi_idx = [i for i in range(self.n_pois)]
        # return a list of the POI objects
        return list(map(POI, x, y, value, time_active, np.ndarray.tolist(n_times_active), slot_size, obs_required, couple, poi_type, n_agents, poi_idx, obs_radius))

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
        self.avg_false = []

    def new_env(self):
        self.pois = self.gen_pois()
        self.agents = self.gen_agents()
        self.avg_false = []

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
            self.pois[j].viewing = []  # if this gets reset at every step, the "viewing" check will only see the last time step

    def run_sim(self, policies, use_time=False):
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
                'number of policies should equal number of agents in system (currently {})'.format(self.n_agents))
        for i in range(len(policies)):
            self.agents[i].policy = policies[i] # sets all agent policies
        for t in range(self.time_steps):
            actions = []
            for agent in self.agents:
                self.state(agent, use_time)  # calculates the state
                act_array = agent.policy(agent.state).detach().numpy()  # picks an action based on the policy
                act = self.action(agent, act_array)
                actions.append(act)  # save the action to list of actions

            self.step(actions)
            # self.avg_false.append(actions.count(False) / len(actions))
            if self.visualize:
                self.draw(t)
        return self.G(), self.D()

    def state(self, agent, use_time=False):
        """
        Takes in an agent, returns the state and the relevant indices for the closest POI or agent in each region of each type
        :param agent:
        :return state, state_idx:
        """
        # Number of sensor bins as rows, poi types plus agents types for columns
        n_agent_types = self.n_agent_types
        if use_time:
            state = np.zeros((self.n_regions, (len(self.poi_options) * 2) + n_agent_types)) - 1
        else:
            state = np.zeros((self.n_regions, (len(self.poi_options)) + n_agent_types)) - 1
        state_idx = np.zeros_like(state) - 1
        poi_dist, poi_quads = self._get_quadrant_state_info(agent, 'p')
        ag_dist, ag_quads = self._get_quadrant_state_info(agent, 'a')
        # Determine closest POI in each region

        for i in range(len(self.pois)):
            d = poi_dist[i]
            if d == -1:         # If the POI is out of range, skip it
                continue        # -1 was used as the arbitrary flag to indicate this POI is out of sensor range
            quad = poi_quads[i]  # Distance portion of the state
            poi_type = self.pois[i].poi_type
            type_2 = poi_type + len(self.poi_options)  # completeness / timing state
            # d is inverse distance, so this finds the closest one
            if d > state[quad, poi_type]:
                state[quad, poi_type] = d
                state_idx[quad, poi_type] = i
                if use_time:
                    state[quad, type_2] = self.pois[i].percent_complete

        # Determine the closest agent in each region and sum inverse distances of all agents in each quadrant
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

    def joint_state(self):
        return [self.state(agent) for agent in self.agents]

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
            num_points = self.n_pois
            points = self.pois
        # Info for other agents
        else:
            num_points = self.n_agents
            points = self.agents
        dist_arr = np.zeros(num_points)  # distance to each POI
        theta_arr = np.zeros(num_points)  # angle to each poi [-pi, pi]
        for i in range(num_points):
            # Compare each point (POI or other agent) to position of current agent
            # Check to make sure it is not the same agent happens later (ignore for now for indexing reasons)
            point = points[i]
            if not point.active:
                dist_arr[i] = -1
                continue
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
        # Digitize is SLOOOOOOWWWWWWWWW
        # dumb_quadrants = np.digitize(theta_arr, bins=self.sensor_bins) - 1  # which quadrant / octant each POI is in relative to the GLOBAL frame
        quadrants = self.n_regions - np.searchsorted(-self.sensor_bins, theta_arr)
        # If it's perfectly on the border between quads 0 & 7, it will cause index issues
        for x in range(len(quadrants)):
            if quadrants[x] == 8:
                quadrants[x] = 7
        return dist_arr, quadrants

    def state_size(self, use_time):
        """
        state size is the discretization of the sensor (number of bins) times the number of POI types plus one
        In each region, the sensor will have one bit for each POI type (distance) and one bit for the sum of the inverse distance to all other agents in that region
        :return:
        state size
        """
        if use_time:
            return self.n_regions * (self.n_poi_types*2 + self.n_agent_types)
        else:
            return self.n_regions * (self.n_poi_types + self.n_agent_types)

    def action(self, agent, nn_output):
        """

        :param agent:
        :param nn_output: Assumes output from the NN a 1xN numpy array
        :return: agent, poi, or False (for no movement)
        """
        nn_max_idx = np.argmax(nn_output)

        # first (n_regions) number of outputs represent POIs
        if nn_max_idx < self.n_regions:
            # Get the index of the max value (aka min distance) POI in that region of any type
            state_idx = np.argmax(agent.state[nn_max_idx][:self.n_poi_types])

            if np.max(agent.state[nn_max_idx][state_idx]) == -1:
                # Flag that there's nothing of that type in that region
                theta = (self.sensor_bins[nn_max_idx] + self.sensor_bins[nn_max_idx+1]) / 2
                # Create a dummy "POI" that has x & y values for the agent to move toward
                region = FalsePOI(agent.x, agent.y, theta, self.size)
                return region

            # Find and return the closest POI in that region
            poi_idx = int(agent.state_idx[nn_max_idx][state_idx])
            return self.pois[poi_idx]

        # Second (n_regions) number of outputs represent agents
        elif nn_max_idx < self.n_regions * 2:

            if not self.with_agents:
                return False

            # Need to get the region number (subtract n_reigons since this is the second set of them)
            nn_max_idx -= self.n_regions
            # Get the index the max value (aka min distance) set of agents in that region of any type
            state_idx = np.argmax(agent.state[nn_max_idx][self.n_poi_types:]) + self.n_poi_types

            if np.max(agent.state[nn_max_idx][state_idx]) == -1:
                # Flag that there's nothing of that type in that region
                theta = (self.sensor_bins[nn_max_idx] + self.sensor_bins[nn_max_idx+1]) / 2
                # Create a dummy "POI" that has x & y values for the agent to move toward
                region = FalsePOI(agent.x, agent.y, theta, self.size)
                return region

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
    from teaming.parameters00 import Parameters
    np.random.seed(0)
    poi_types = [[100, 1, 0]]  # , [10, 3, 1], [50, 5, 2]]
    num_agents = 1
    num_pois = 1
    num_states = 2 ** (num_pois * 2)
    param = Parameters()
    env = DiscreteRoverDomain(param)

    for i in range(30):

        actions = [env.pois[0]]
        env.step(actions)

        # env.draw()
        print(i, env.G(), env.D())
    for agent in env.agents:
        print(env.state(agent))
