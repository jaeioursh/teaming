import numpy as np


class Agent:
    def __init__(self, x, y, idx, cap, type, p):
        self.class_type = 'Agent'
        self.idx = idx
        self.x = x                  # location - x
        self.y = y                  # location - y
        self._x = x                 # Initial location
        self._y = y                 # Initial location
        self.p = p
        self.curr_rm = None
        self.type = type            # Agent type - TODO: this should not be hard-coded
        self.xy_goal = None             # variable to store desired goal
        self.poi = None
        # self.capabilities = cap     # randomly initialize agent's capability of viewing each POI
        self.capabilities = np.ones_like(cap)       # Agents are equally good at finding all POIs
        self.policy = None
        self.state = None           # Current state
        self.rm_timers = np.zeros(len(self.p.rooms) + 1) + 100   # Keep track of how long it has been since last in each room - everything starts as 'never been'
        self.time_in_rm = np.zeros(len(self.p.rooms) + 1)
        self.rm_in_state = np.zeros_like(self.rm_timers)    # Binary to determine whether info drops out of state (0 don't include / 1 include)

    def reset(self):
        self.xy_goal = None             # reset to no desired POI
        self.poi = None
        self.curr_rm = None
        self.policy = None
        self.state = None
        self.x = self._x                # Reset to initial location
        self.y = self._y

        self.rm_timers = np.zeros(len(self.p.rooms) + 1) + 100  # Keep track of how long it has been since last in each room - everything starts as 'never been'
        self.time_in_rm = np.zeros(len(self.p.rooms) + 1)
        self.rm_in_state = np.zeros_like(self.rm_timers)  # Binary to determine whether info drops out of state (0 don't include / 1 include)

    def step(self):
        if self.xy_goal:
            self.move()                     # move agent toward POI
            rew = self.observe()
            if rew:                         # If at the POI and observed
                poi = self.poi              # get the POI
                poi.viewing.append(self)    # add the agent to current agents viewing the POI
                poi.viewed.append(self)
                # poi.viewed_rew.append(rew)
                self.xy_goal = None

    # moves agent 1-unit towards the POI
    def move(self):
        """
        If the agent has a desired POI, move one unit toward POI
        :return:
        """
        goal_exists = [x for x in self.xy_goal if x]        # This checks that something other than None is in the array
        if goal_exists:
            X = self.xy_goal[0]
            Y = self.xy_goal[1]
            try:
                R = ((X-self.x)**2.0+(Y-self.y)**2.0)**0.5
            except TypeError:
                print(X, Y, self.x, self.y)
            if R > 1:
                self.y += (Y-self.y)/R
                self.x += (X-self.x)/R
            else:
                self.y += (Y-self.y)
                self.x += (X-self.x)

    # boolean to check if agent is successful in observing desired POI
    def observe(self):
        """
        If agent is within the observation radius, it is successful in observing
        :return:
        """
        # If there is no POI,
        if not self.poi:
            return 0
        elif abs(self.poi.x - self.x) <= self.poi.obs_radius and abs(self.poi.y - self.y) <= self.poi.obs_radius:
            return self.poi.value
        else:
            return 0

    def update_rm_st(self):
        # Add one to each room timer
        self.rm_timers += 1
        # Set current room to zero
        self.rm_timers[self.curr_rm] = 0
        self.time_in_rm[self.curr_rm] += 1
        # Set the boolean value to 0 if greater than the time threshold, otherwise 1
        self.rm_in_state = np.zeros_like(self.rm_timers)
        self.rm_in_state[self.rm_timers < self.p.time_threshold] = 1
