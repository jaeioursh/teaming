import numpy as np
from math import cos, sin


class POI:
    def __init__(self, x, y, value, refresh_rate, obs_required, couple, poi_type, n_agents, poi_idx, obs_radius,
                 strong_coupling=False):
        self.class_type = 'POI'
        self.x = x                          # location - x
        self.y = y                          # location - y
        self.value = value                  # POI value -- this only makes sense for some reward structures
        self.refresh_rate = refresh_rate    # how often it is refreshed
        self.obs_required = obs_required    # number of observations required to fully observe the POI
        self.couple = couple                # coupling requirement
        self.poi_type = poi_type            # type
        self.poi_idx = poi_idx              # ID for each POI
        self.strong_coupling = strong_coupling      # 1: simultaneous observation,  0: observations within window of time
        self.obs_radius = obs_radius        # observation radius

        self.refresh_idx = 0                # time steps since last refresh
        self.curr_rew = 0                   # Current reward will allow the agents to get a local reward when this is observed
        self.successes = 0                  # number of times it has successfully been captured
        self.observed = 0                   # 0: not observed during this refresh cycle | 1: observed during this cycle
        self.claimed = False

        self.D_vec = np.zeros(n_agents)     # difference rewards
        self.Dpp_vec = np.zeros(n_agents)   # D++ rewards
        self.viewed = []                    # list of all agents that viewed in refresh window
        self.viewing = []                   # list of currently observing agents
        self.history = []                   # history of agents that have viewed this POI

    def reset(self):
        """
        Reset refresh, successes, viewed, and viewing
        :return:
        """
        self.refresh_idx = 0
        self.curr_rew = 0
        self.successes = 0
        self.observed = 0
        self.D_vec[:] = 0
        self.Dpp_vec[:] = 0
        self.viewed = []
        self.viewing = []
        self.history = []

    def refresh(self):
        self.refresh_idx += 1  # increase number of time steps since last refresh
        if not self.observed: # if it has not yet been observed this refresh cycle
            if self.strong_coupling:  # if it requires simultaneous observation
                if len(self.viewing) > 0:
                    idxs = [agent.idx for agent in self.viewing]
                    capabilities = [agent.capabilities[self.poi_type] for agent in self.viewing]  # check to make sure the agents are capable of observing this POI
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
                                if agent.capabilities[self.poi_type] == g:
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
            else:
                if len(self.viewed) >= self.obs_required:
                    # This is necessary for the greedy base comparison policy
                    self.observed = 1
        if self.refresh_idx == self.refresh_rate:  # if it has hit the refresh
            if not self.strong_coupling:
                if len(self.viewed) >= self.obs_required:  # if weak coupling, check all the agents that viewed this refresh cycle
                        # NOTE: This currently assumes agents have uniform (all ones) capabilities.
                        # If you need to use this with heterogeneous agents, it will need to be amended.
                        self.successes += 1

                        idxs = [agent.idx for agent in self.viewed]
                        unique = np.unique(idxs)
                        ag_d = np.zeros_like(unique)
                        for i, ag in enumerate(unique):
                            ag_d[i] = 0
                            temp_arr = [idx for idx in idxs if idx != ag]
                            if len(temp_arr) < self.obs_required:
                                # If the observation is not met without this agent, then it gets the full value of the POI
                                ag_d[i] = self.value
                        self.D_vec[unique] += np.array(ag_d)

            self.refresh_idx = 0  # reset the time steps
            self.observed = 0
            self.viewed = []
            self.claimed = False



class FalsePOI:
    def __init__(self, ag_x, ag_y, theta, bounds):
        self.class_type = "Region"
        self.x = ag_x + (1.01 * cos(theta))
        self.y = ag_y + (1.01 * sin(theta))
        self.check_bounds(bounds)

    def check_bounds(self, bounds):
        if self.x < 0:
            self.x = 0
        elif self.x > bounds:
            self.x = bounds
        if self.y < 0:
            self.y = 0
        elif self.y > bounds:
            self.y = bounds
