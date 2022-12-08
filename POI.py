import numpy as np


class POI:
    def __init__(self, poi_idx, x, y, p, poi_type, rm_idx):
        self.poi_idx = poi_idx              # ID for each POI
        self.x = x                          # location - x
        self.y = y                          # location - y
        self.p = p                          # Global parameters
        self.type = poi_type            # type (numerical type to keep track of the same POI types)
        self.rm_idx = rm_idx                # Keep track of which room this POI is in

        self.couple = self.p.couple                # coupling requirement
        self.obs_radius = self.p.obs_radius        # observation radius
        self.strong_coupling = self.p.strong_coupling      # 1: simultaneous observation,  0: observations within window of time

        self.value = 1                      # Value of POI
        self.class_type = 'POI'
        self.successes = 0                  # number of times it has successfully been captured
        self.observed = 0                   # 0: not observed during this refresh cycle | 1: observed during this cycle
        self.claimed = 0                    # Used for greedy omniscient policy
        self.D_vec = np.zeros(p.n_agents)     # difference rewards
        self.Dpp_vec = np.zeros(p.n_agents)   # D++ rewards
        self.viewed = []                    # list of all agents that viewed this POI
        self.viewing = []                   # list of currently observing agents

    def reset(self):
        """
        Reset refresh, successes, viewed, and viewing
        :return:
        """
        self.D_vec = np.zeros(self.p.n_agents)     # difference rewards
        self.Dpp_vec = np.zeros(self.p.n_agents)   # D++ rewards
        self.successes = 0                  # number of times it has successfully been captured
        self.observed = 0                   # 0: not observed during this refresh cycle | 1: observed during this cycle
        self.claimed = 0                    # Used for greedy omniscient policy
        self.viewed = []                    # list of all agents that viewed in refresh window
        self.viewing = []                   # list of currently observing agents

    def refresh(self):
        if len(self.viewed) >= self.couple:  # if weak coupling, check all the agents that viewed this refresh cycle
            self.observed = 1
            self.successes = 1

    def refresh_weak(self):
        # TODO: put in logic for D / D++ Rewards
        pass

    def refresh_strong(self):
        if len(self.viewing) >= self.couple:  # if weak coupling, check all the agents that viewed this refresh cycle
            self.observed = 1
        else:
            self.observed = 0

        if self.observed:  # if it has not yet been observed this refresh cycle
            if len(self.viewing) > 0:
                idxs = [agent.idx for agent in self.viewing]
                capabilities = [agent.capabilities[self.type] for agent in
                                self.viewing]  # check to make sure the agents are capable of observing this POI
                capabilities = sorted(capabilities)
                g = capabilities[0]  # Add minimum capability of agents to success of observation
                if len(self.viewing) >= self.couple:  # if the number of simultaneous observations meets the coupling requirement
                    self.observed = 1
                    self.successes += g
                    # difference reward block
                    d = []
                    dpp = []
                    for agent in self.viewing:
                        if len(self.viewing) >= self.couple + 1:  # too many viewing
                            if agent.capabilities[self.type] == g:
                                d.append(g - capabilities[1])
                            else:
                                d.append(0)
                        else:  # exactly the right number viewing
                            d.append(g)

                    self.D_vec[idxs] += np.array(d)
                    self.Dpp_vec[idxs] += np.array(d)
                else:  # not enough observing
                    n_needed = self.couple - len(self.viewing)
                    dpp = [g / n_needed] * len(self.viewing)
                    self.Dpp_vec[idxs] += np.array(dpp)
