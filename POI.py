import numpy as np
from math import cos, sin, pi
from scipy import signal    # For square wave

class POI:
    def __init__(self, x, y, couple, poi_type, str_type, n_agents, poi_idx, obs_radius, tot_time, rand_shift=None, strong_coupling=False):
        self.class_type = 'POI'
        self.x = x                          # location - x
        self.y = y                          # location - y
        self.couple = couple                # coupling requirement
        self.poi_type = poi_type            # type
        self.str_type = str_type
        self.poi_idx = poi_idx              # ID for each POI
        self.strong_coupling = strong_coupling      # 1: simultaneous observation,  0: observations within window of time
        self.obs_radius = obs_radius        # observation radius
        self.tot_time = tot_time
        self.n_agents = n_agents
        self.value = 1

        self.time_rewards = np.zeros(tot_time)
        self.active = 0
        self.curr_time = 0                  # Keep track of the current time step
        self.curr_rew = 0                   # Current reward will allow the agents to get a local reward when this is observed
        self.successes = 0                  # number of times it has successfully been captured
        self.observed = 0                   # 0: not observed during this refresh cycle | 1: observed during this cycle
        self.claimed = 0                    # Used for greedy omniscient policy
        self.rand_shift = rand_shift                 # Shift of the reward function
        self.D_vec = np.zeros(n_agents)     # difference rewards
        self.Dpp_vec = np.zeros(n_agents)   # D++ rewards
        self.viewed = []                    # list of all agents that viewed in refresh window
        self.viewed_rew = []
        self.viewing = []                   # list of currently observing agents
        self.history = []                   # history of agents that have viewed this POI

        self.setup_times()
        self.set_active()                   # Sets current reward and active status

    def reset(self):
        """
        Reset refresh, successes, viewed, and viewing
        :return:
        """
        self.D_vec = np.zeros(self.n_agents)     # difference rewards
        self.Dpp_vec = np.zeros(self.n_agents)   # D++ rewards
        self.active = 0
        self.curr_time = 0                  # Keep track of the current time step
        self.curr_rew = 0                   # Current reward will allow the agents to get a local reward when this is observed
        self.successes = 0                  # number of times it has successfully been captured
        self.observed = 0                   # 0: not observed during this refresh cycle | 1: observed during this cycle
        self.claimed = 0                    # Used for greedy omniscient policy
        self.viewed = []                    # list of all agents that viewed in refresh window
        self.viewed_rew = []
        self.viewing = []                   # list of currently observing agents
        self.history = []                   # history of agents that have viewed this POI
        self.set_active()                   # Sets current reward and active status

    def setup_times(self):
        x = np.linspace(0, self.tot_time - 1, self.tot_time)
        if self.str_type == 'sin' or self.str_type == 'cos':
            # This function gives you precisely two active waves during [0,60]
            if self.rand_shift is None:
                self.rand_shift = np.random.uniform(0, pi)
            self.time_rewards = .5 * (1 - np.cos((.21 * x) - self.rand_shift))
        elif self.str_type == 'exp':
            # This function provides exponential decay that drops below 0.1 at around 20 time steps
            # if self.rand_shift is None:
            self.rand_shift = np.random.choice([-1, 1])
            # the rand shift either keeps the original shape or reverses it to be exponential growth (between [0,1])
            self.time_rewards = np.exp(-0.1 * x)[::self.rand_shift]
        elif self.str_type == 'sq' or self.str_type == 'square':
            # This provides two square waves, the second of which has half the amplitude of the first
            if self.rand_shift is None:
                self.rand_shift = np.random.uniform(-1, 1)
            out_array = .5 * (1 - signal.square(.2 * x + self.rand_shift))
            # Array that reduces the values in the second half of the array by 50%
            mult_array = np.ones(self.tot_time)
            mid = int(self.tot_time / 2) - 2
            mult_array[-mid:] *= 0.5
            self.time_rewards = out_array * mult_array
        elif self.str_type == 'on':
            self.time_rewards = np.ones(self.tot_time)
        else:
            raise ValueError(f"POI type {self.str_type} not recognized")

    def set_active(self):
        # Set the current reward and active status
        if self.curr_time >= self.tot_time:
            return

        self.curr_rew = self.time_rewards[self.curr_time]
        if self.curr_rew < 0.01:
            self.active = 0
        else:
            self.active = 1

    def refresh(self):
        self.curr_time += 1  # increase number of time steps since last refresh
        self.set_active()    # Sets current reward and active status
        # if self.strong_coupling:
        #     self.refresh_strong()
        if self.curr_time == self.tot_time:
            self.refresh_weak()
        if len(self.viewed) >= self.couple:  # if weak coupling, check all the agents that viewed this refresh cycle
            self.observed = 1

    def refresh_weak(self):
        if len(self.viewed) >= self.couple:  # if weak coupling, check all the agents that viewed this refresh cycle
            # TODO: Change this to work for coupling, because right now it doesn't
            # BIGGER NOTE: This currently ONLY rewards the best agent. because I'm not working with coupling right now
            # NOTE: This currently assumes agents have uniform (all ones) capabilities.
            # If you need to use this with heterogeneous agents, it will need to be amended.
            idxs = [agent.idx for agent in self.viewed]
            unique = np.unique(idxs)
            ag_d = np.zeros_like(unique, dtype=float)

            # Find the two biggest values
            sort_rew = np.argsort(self.viewed_rew)
            max_idx = sort_rew[-2:]

            # Subtract second largest from the largest captured reward
            if len(max_idx) > 1:
                best_ag = np.where(unique == idxs[max_idx[1]])[0][0]
                second_ag = np.where(unique == idxs[max_idx[0]])[0][0]
                if best_ag == second_ag:
                    # If the best agent also got the next best score, just give it to them
                    max_d = self.viewed_rew[max_idx[1]]
                else:
                    # Otherwise subtract off the next best score
                    max_d = self.viewed_rew[max_idx[1]] - self.viewed_rew[max_idx[0]]
            else:
                # If only one agent visited, then they get that score
                best_ag = np.where(unique == idxs[max_idx[0]])[0][0]
                max_d = self.viewed_rew[max_idx[0]]
            # This finds the index of the best agent in the UNIQUE array
            # best_ag = np.where(unique == max_idx[1])[0][0]
            ag_d[best_ag] = max_d

            self.D_vec[unique] += ag_d
            self.viewed = []
            self.viewed_rew = []
            # These are necessary for the greedy base comparison policy
            self.claimed = 0
            self.observed = 1
            self.successes += max_d

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
