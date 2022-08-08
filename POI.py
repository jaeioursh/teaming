import numpy as np
from math import cos, sin


class POI:
    def __init__(self, x, y, value, time_active, n_times_active, time_slots, obs_required, couple, poi_type, n_agents, poi_idx, obs_radius,
                 strong_coupling=False):
        self.class_type = 'POI'
        self.x = x                          # location - x
        self.y = y                          # location - y
        self.value = value                  # POI value -- this only makes sense for some reward structures
        self.obs_required = obs_required    # number of observations required to fully observe the POI
        self.couple = couple                # coupling requirement
        self.poi_type = poi_type            # type
        self.poi_idx = poi_idx              # ID for each POI
        self.strong_coupling = strong_coupling      # 1: simultaneous observation,  0: observations within window of time
        self.obs_radius = obs_radius        # observation radius
        self.times_active = []
        self.active = False

        self.curr_time = 0                # time steps since last refresh
        self.curr_rew = 0                   # Current reward will allow the agents to get a local reward when this is observed
        self.successes = 0                  # number of times it has successfully been captured
        self.observed = 0                   # 0: not observed during this refresh cycle | 1: observed during this cycle
        self.claimed = 0

        self.D_vec = np.zeros(n_agents)     # difference rewards
        self.Dpp_vec = np.zeros(n_agents)   # D++ rewards
        self.viewed = []                    # list of all agents that viewed in refresh window
        self.viewing = []                   # list of currently observing agents
        self.history = []                   # history of agents that have viewed this POI

        self.setup_times(time_active, n_times_active, time_slots)

    def reset(self):
        """
        Reset refresh, successes, viewed, and viewing
        :return:
        """
        self.curr_time = 0
        self.curr_rew = 0
        self.successes = 0
        self.observed = 0
        self.D_vec[:] = 0
        self.Dpp_vec[:] = 0
        self.viewed = []
        self.viewing = []
        self.history = []
        self.claimed = 0
        if self.times_active[0][0] == 0:
            self.active = True
        else:
            self.active = False

    def setup_times(self, time, num_times, slots):
        for i in range(int(num_times)):
            if time == slots:
                beginning = i * time
                end = ((i + 1) * time) - 1
            else:
                slot0 = i * slots
                slot1 = ((i+1) * slots) - time
                beginning = np.random.randint(slot0, slot1 + 1)
                end = beginning + time - 1
            if beginning == 0:
                self.active = True
            self.times_active.append([beginning, end])

    def refresh(self):
        self.curr_time += 1  # increase number of time steps since last refresh
        if self.strong_coupling:
            self.refresh_strong()
        else:
            self.refresh_weak()

    def refresh_weak(self):
        if len(self.viewed) >= self.obs_required:  # if weak coupling, check all the agents that viewed this refresh cycle
            # This is necessary for the greedy base comparison policy
            self.observed = 1
        for [st, end] in self.times_active:
            if self.curr_time == end:  # if it has hit the refresh
                self.active = False
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
                            ag_d[i] = 1
                    self.D_vec[unique] += np.array(ag_d)
                self.observed = 0
                self.viewed = []
                self.claimed = 0
            if self.curr_time == st:
                self.active = True

    def refresh_strong(self):
        if not self.observed:  # if it has not yet been observed this refresh cycle
            if len(self.viewing) > 0:
                idxs = [agent.idx for agent in self.viewing]
                capabilities = [agent.capabilities[self.poi_type] for agent in
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
