import numpy as np

class Agent:
    def __init__(self, x, y, cap,idx):
        self.idx=idx
        self.x = x                  # location - x
        self.y = y                  # location - y
        self._x = x                 # initial location - x
        self._y = y                 # initial location - y
        self.poi = None             # variable to store desired POI
        self.capabilities = cap    # randomly initialize agent's capability of viewing each POI

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
            R = ((X-self.x)**2.0+(Y-self.y)**2.0)**0.5
            if R>1:
                self.y+=(Y-self.y)/R
                self.x+=(X-self.x)/R
            
            

    # boolean to check if agent is successful in observing desired POI
    def observe(self):
        """
        If agent is within the observation radius, it is successful in observing
        :return:
        """
        if abs(self.poi.x - self.x) < self.poi.obs_radius and abs(self.poi.y - self.y) < self.poi.obs_radius:
            return 1
        else:
            return 0
