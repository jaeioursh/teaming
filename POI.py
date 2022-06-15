import numpy as np

class POI:
    def __init__(self, x, y, value, refresh_rate, obs_required, couple, poi_type,n_agents, strong_coupling=True):
        self.value = value                  # POI value -- this only makes sense for some reward structures
        self.successes = 0                  # number of times it has successfully been captured
        self.D_vec=np.zeros(n_agents)
        self.Dpp_vec=np.zeros(n_agents)
        self.refresh_idx = 0                # time steps since last refresh
        self.refresh_rate = refresh_rate    # how often it is refreshed
        self.obs_required = obs_required    # number of observations required to fully observe the POI
        self.obs_radius = 2                 # observation radius
        self.couple = couple                # coupling requirement
        self.x = x                          # location - x
        self.y = y                          # location - y
        self.poi_type = poi_type            # type
        self.strong_coupling = strong_coupling  # 1: simultaneous obvservation,  0: observations within window of time
        self.viewed = []                    # list of all agents that viewed in refresh window
        self.viewing = []                   # list of currently observing agents


    def reset(self):
        """
        Reset refresh, successes, viewed, and viewing
        :return:
        """
        self.refresh_idx = 0
        self.successes = 0
        self.D_vec[:]=0
        self.Dpp_vec[:]=0
        self.viewed = []
        self.viewing = []

    def refresh(self):
        self.refresh_idx += 1                               # increase number of time steps since last refresh
        if self.refresh_idx == self.refresh_rate:           # if it has hit the refresh
            self.refresh_idx = 0                            # reset the time steps
            if self.strong_coupling:                        # if it requires simultaneous observation
                if len(self.viewing)>0:
                    idxs=[agent.idx for agent in self.viewing]
                    capabilities = [agent.capabilities[self.poi_type] for agent in self.viewing]    # check to make sure the agents are capable of observing this POI
                    capabilities=sorted(capabilities)
                    g=capabilities[0]     # Add minimum capability of agents to success of observation
                    if len(self.viewing) >= self.couple:        # if the number of simultaneous observations eets the coupling requirement
                        self.successes += g
                        #difference reward block
                        d=[]
                        dpp=[]
                        for agent in self.viewing:
                            if len(self.viewing)>=self.couple+1:  # too many viewing
                                if agent.capabilities[self.poi_type]==g:
                                    d.append(g-capabilities[1])
                                else:
                                    d.append(0)
                            else: #exactly the right number viewing
                                d.append(g)
                                            
                        self.D_vec[idxs]+=np.array(d)
                        self.Dpp_vec[idxs]+=np.array(d)
                    else: #not enough observing
                        n_needed=self.couple-len(self.viewing)
                        dpp=[g/n_needed] * len(self.viewing)
                        self.Dpp_vec[idxs]+=np.array(dpp)
   
            else:
                if len(self.viewed) >= self.couple:         # if weak coupling, check all the agents that viewed this refresh cycle
                    capabilities = [agent.capabilities[self.poi_type] for agent in self.viewed]
                    g=min(capabilities)
                    self.successes += g

            self.viewed = []                                # reset the agents that have viewed

