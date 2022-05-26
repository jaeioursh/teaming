import numpy as np


class POI:
    def __init__(self, x, y, value, refresh_rate, obs_required, couple, poi_type, poi_idx, strong_coupling=False):
        self.x = x                          # location - x
        self.y = y                          # location - y
        self.value = value                  # POI value -- this only makes sense for some reward structures
        self.refresh_rate = refresh_rate    # how often it is refreshed
        self.refresh_idx = 0                # time steps since last refresh
        self.obs_required = obs_required    # number of observations required to fully observe the POI
        self.obs_radius = 2                 # observation radius
        self.couple = couple                # coupling requirement
        self.poi_type = poi_type            # type
        self.poi_idx = poi_idx              # ID for each POI
        self.successes = 0                  # number of times it has successfully been captured
        self.strong_coupling = strong_coupling  # 1: simultaneous obvservation,  0: observations within window of time
        self.viewed = []                    # list of all agents that viewed in refresh window
        self.viewing = []                   # list of currently observing agents
        self.history = []                   # history of agents that have viewed this POI

    def reset(self):
        """
        Reset refresh, successes, viewed, and viewing
        :return:
        """
        self.refresh_idx = 0
        self.successes = 0
        self.viewed = []
        self.viewing = []

    def refresh(self):
        self.refresh_idx += 1                               # increase number of time steps since last refresh
        if self.refresh_idx == self.refresh_rate:           # if it has hit the refresh
            self.refresh_idx = 0                            # reset the time steps
            if self.strong_coupling:                        # if it requires simultaneous observation
                if len(self.viewing) >= self.couple:        # if the number of simultaneous observations meets the coupling requirement
                    capabilities = [agent.capabilities[self.poi_type] for agent in self.viewing]    # check to make sure the agents are capable of observing this POI
                    self.successes += min(capabilities)     # Add minimum capability of agents to success of observation
                    self.history.append(self.viewing)       # History of agents that simultaneously viewed the POI
            else:
                if len(self.viewed) >= self.obs_required:   # if weak coupling, check all the agents that viewed this refresh cycle
                    capabilities = [agent.capabilities[self.poi_type] for agent in self.viewed]
                    self.successes += min(capabilities)
                    self.history.append(self.viewed)
            self.viewed = []                                # reset the agents that have viewed


class Agent:
    def __init__(self, x, y, N_pois):
        self.x = x                  # location - x
        self.y = y                  # location - y
        self._x = x                 # initial location - x
        self._y = y                 # initial location - y
        self.poi = None             # variable to store desired POI
        self.capabilities = np.ones(N_pois)    # randomly initialize agent's capability of viewing each POI

    def reset(self):
        self.x = self._x            # magically teleport to initial location
        self.y = self._y            # magically teleport to initial location
        self.poi = None             # reset to no desired POI

    # moves agent 1-unit towards the POI
    def move(self):
        """
        If the agent has a desired POI, move one unit toward POI
        :return:
        """
        if self.poi is not None:
            X = self.poi.x
            Y = self.poi.y
            if X > self.x:
                self.x += 1
            elif X < self.x:
                self.x -= 1
            elif Y > self.y:
                self.y += 1
            elif Y < self.y:
                self.y -= 1

    # boolean to check if agent is successful in observing desired POI
    def observe(self):
        """
        If agent is within the observation radius, it is successful in observing
        :return:
        """
        if abs(self.poi.x - self.x) < self.poi.couple and abs(self.poi.y - self.y) < self.poi.couple:
            return 1
        else:
            return 0


class DiscreteRoverDomain:
    def __init__(self, N_agents, N_pois, poi_options):
        self.N_agents = N_agents            # number of agents
        self.N_pois = N_pois                # number of POIs
        self.poi_options = np.array(poi_options)      # options for each POI - [refresh rate, number of observations required, ID]

        self.size = 30                      # size of the world

        self.agents = self.gen_agents()     # generate agents
        self.pois = self.gen_pois()         # generate POIs
        self.reset()                        # reset the system

    # generate list of agents
    def gen_agents(self):
        """
        Generates a list of agents
        :return: list of agent objects at random initial locations
        """
        # creates an array of x, y positions for each agent
        # locations are [0,4] plus half the size of the world
        self.starting_locs = np.random.randint(0, 4, (self.N_agents, 2)) + self.size // 2
        # return an array of Agent objects at the specified locations
        return [Agent(x, y, self.N_pois) for x, y in self.starting_locs]

    # generate list of POIs
    def gen_pois(self):
        """
        Generates a list of random POIs
        :return: list of POI objects
        """
        num_poi_types = np.shape(self.poi_options)[0]
        randints = np.random.randint(num_poi_types, size=self.N_pois)     # get an array of random integers - integers represent the POI type, one for each POI in the world
        poi_vals = self.poi_options[randints, :]                     # array of values from the POI options for each POI in the world
        x = np.random.randint(0, self.size, self.N_pois)        # x locations for all POIs
        y = np.random.randint(0, self.size, self.N_pois)        # y locations for all POIs

        # refresh_rate = [10 for i in range(self.N_pois)]         # refresh rate for all POIs
        # obs_required = [2 for i in range(self.N_pois)]          # number of observations required
        # poi_type = [i for i in range(self.N_pois)]              # Each one is a different type

        refresh_rate = np.ndarray.tolist(poi_vals[:, 0])
        obs_required = np.ndarray.tolist(poi_vals[:, 1])
        poi_type = np.ndarray.tolist(poi_vals[:, 2])

        couple = [2 for _ in range(self.N_pois)]                # coupling requirement for all POIs
        value = [1 for _ in range(self.N_pois)]                 # Value of the POI
        poi_idx = [i for i in range(self.N_pois)]
        # return a list of the POI objects
        return list(map(POI, x, y, value, refresh_rate, obs_required, couple, poi_type, poi_idx))

    # reset environment to intial config
    def reset(self):
        """
        Reset environment to initial configuration
        :return:
        """
        for a in self.agents:       # reset all agents to initial config
            a.reset()
        for p in self.pois:         # reset all POIs to initial config
            p.reset()

    # perform one state transition given a list of actions for each agent
    def step(self, actions):
        """
        perform one state transition given a list of actions for each agent
        :param actions:
        :return:
        """
        # update all agents
        for i in range(self.N_agents):
            if not self.agents[i].poi:
                self.agents[i].poi = self.pois[actions[i]]  # agents set a new goal at every time step
            self.agents[i].move()                       # move agent toward POI
            if self.agents[i].observe():                # If at the POI and observed
                poi = self.agents[i].poi                # get the POI ID
                poi.viewing.append(self.agents[i])      # add the agent to current agents viewing the POI
                poi.viewed.append(self.agents[i])
                self.agents[i].poi = None
        # refresh all POIs and reset which agents are currently viewing
        for i in range(self.N_pois):
            self.pois[i].refresh()
            self.pois[i].viewing = []                   # if this gets reset at every step, the "viewing" check will only see the last time step

    # TODO
    def state(self):
        pass

    # returns global reward based on POI values
    def G(self):
        g = 0
        for poi in self.pois:
            g += poi.successes * poi.value
        return g


if __name__ == "__main__":
    np.random.seed(0)
    # POI types should be of the form [refresh rate, number of observations required, unique ID]
    poi_types = [[1, 1, 0], [10, 3, 1], [50, 5, 2]]
    env = DiscreteRoverDomain(3, 3, poi_types)

    for i in range(1000):
        actions = np.ndarray.tolist(np.random.randint(3, size=3))
        # actions = [1, 3, 4]
        env.step(actions)
        print(i, env.G())
