import numpy as np


class Agent:
    def __init__(self, x, y, N_pois):
        self.x = x                  # location - x
        self.y = y                  # location - y
        self._x = x                 # initial location - x
        self._y = y                 # initial location - y
        self.poi = None             # variable to store desired POI
        self.last_visit = np.zeros(N_pois)      # array to keep track of the last time this agent visited each POI
        self.capabilities = np.ones(N_pois)     # randomly initialize agent's capability of viewing each POI
        self.policy = None

    def reset(self):
        self.x = self._x            # magically teleport to initial location
        self.y = self._y            # magically teleport to initial location
        self.poi = None             # reset to no desired POI

    def step(self, new_poi):
        if not self.poi:
            self.poi = new_poi          # If the agent isn't already on the way to a POI
        self.move()                     # move agent toward POI
        self.last_visit += 1
        if self.observe():              # If at the POI and observed
            poi = self.poi              # get the POI
            poi.viewing.append(self)    # add the agent to current agents viewing the POI
            poi.viewed.append(self)
            self.last_visit[self.poi.poi_idx] = 0   # reset the time since this agent viewed that POI
            self.poi = None

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
            self.last_visit[self.poi.poi_idx] = 0
            return 1
        else:
            return 0
