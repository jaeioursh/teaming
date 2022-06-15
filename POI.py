
class POI:
    def __init__(self, x, y, value, refresh_rate, obs_required, couple, poi_type, poi_idx, strong_coupling=False):
        self.x = x                          # location - x
        self.y = y                          # location - y
        self.value = value                  # POI value -- this only makes sense for some reward structures
        self.refresh_rate = refresh_rate    # how often it is refreshed
        self.refresh_idx = 0                # time steps since last refresh
        self.curr_rew = 0                   # Current reward will allow the agents to get a local reward when this is observed
        self.obs_required = obs_required    # number of observations required to fully observe the POI
        self.obs_radius = 1                 # observation radius
        self.couple = couple                # coupling requirement
        self.poi_type = poi_type            # type
        self.poi_idx = poi_idx              # ID for each POI
        self.successes = 0                  # number of times it has successfully been captured
        self.strong_coupling = strong_coupling  # 1: simultaneous observation,  0: observations within window of time
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
        self.curr_rew = 0
        if self.refresh_idx == self.refresh_rate:           # if it has hit the refresh
            self.refresh_idx = 0                            # reset the time steps
            if self.strong_coupling:                        # if it requires simultaneous observation
                if len(self.viewing) >= self.couple:        # if the number of simultaneous observations meets the coupling requirement
                    capabilities = [agent.capabilities[self.poi_type] for agent in self.viewing]    # check to make sure the agents are capable of observing this POI
                    self.successes += min(capabilities)     # Add minimum capability of agents to success of observation
                    self.history.append(self.viewing)       # History of agents that simultaneously viewed the POI
                    self.curr_rew = 1
            else:
                if len(self.viewed) >= self.obs_required:   # if weak coupling, check all the agents that viewed this refresh cycle
                    capabilities = [agent.capabilities[self.poi_type] for agent in self.viewed]
                    self.successes += min(capabilities)
                    self.history.append(self.viewed)
                    self.curr_rew = 1
            self.viewed = []                                # reset the agents that have viewed
        self.viewing = []                   # if this gets reset at every step, the "viewing" check will only see the last time step
