import numpy as np
from teaming.POI import FalsePOI
from math import pi
from random import uniform


class Agent:
    def __init__(self, x, y, idx, cap, type, p):
        self.class_type = 'Agent'
        self.idx = idx
        self.x = x                  # location - x
        self.y = y                  # location - y
        self._x = x                 # initial location - x
        self._y = y                 # initial location - y
        self.prior_x = self.x
        self.prior_y = self.y
        self.p = p
        self.type = type            # Agent type - TODO: this should not be hard-coded
        self.num_stationary = 0
        self.poi = None             # variable to store desired goal
        # self.capabilities = cap     # randomly initialize agent's capability of viewing each POI
        self.capabilities = np.ones_like(cap)       # Agents are equally good at finding all POIs
        self.policy = None
        self.state = None           # Current state
        self.state_idx = None       # Metadata about the state

    def reset(self):
        self.x = self._x            # magically teleport to initial location
        self.y = self._y            # magically teleport to initial location
        self.poi = None             # reset to no desired POI
        self.policy = None
        self.state = None
        self.state_idx = None
        self.prior_y = 0
        self.prior_x = 0
        self.num_stationary = 0

    def step(self):
        if self.poi:
            self.move()                     # move agent toward POI
            if self.observe():              # If at the POI and observed
                poi = self.poi              # get the POI
                poi.viewing.append(self)    # add the agent to current agents viewing the POI
                poi.viewed.append(self)
                self.poi.claimed = False
                self.poi = None

    # moves agent 1-unit towards the POI
    def move(self):
        """
        If the agent has a desired POI, move one unit toward POI
        :return:
        """
        self.prior_x = self.x
        self.prior_y = self.y
        if self.poi is not None:
            X = self.poi.x
            Y = self.poi.y
            R = ((X-self.x)**2.0+(Y-self.y)**2.0)**0.5
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

        if self.poi.class_type == 'Agent' or self.poi.class_type == "Region":
            return 0
        if abs(self.poi.x - self.x) <= self.poi.obs_radius and abs(self.poi.y - self.y) <= self.poi.obs_radius:
            return 1
        else:
            return 0
