import numpy as np
from math import sqrt

class Room:
    def __init__(self, idx, p, bounds, door, pois):
        self.idx = idx
        self.p = p
        self.bounds = bounds  # [lower left, upper right] - [[x1, y1], [x2, y2]]
        self.pois = pois
        self.door = None
        if door:
            self.door = [door[0], door[1]]   # Door comes in as a tuple instead of a list so this is a dumb easy fix

        self.agents_in_rm = []

    def reset(self):
        self.agents_in_rm = []

    def in_room(self, agent):
        """
        Checks if the agent is in the room, returns true or false.
        :param agent:
        :return True or False:
        """
        lower = self.bounds[0]
        upper = self.bounds[1]

        if lower[0] < agent.x < upper[0] and lower[1] < agent.y < upper[1]:
            return True
        # If the agent is in the door, it's in the room.
        if self.door:
            dist_to_door = sqrt((agent.x - self.door[0]) ** 2 + (agent.y - self.door[1]) ** 2)
            # KEEP THE 0.1 VALUE THE SAME AS CHECK IN DOMAIN.ACTION!!! Otherwise shit breaks.
            if dist_to_door < 0.1:
                return True
        return False

    def poi_state(self):
        """
        Returns a list of the number of POIs of each type in this room.
        :return state:
        """
        # Doing it this way instead of numpy zeros so I can use extend (no simple numpy equivalent)
        st = [0] * self.p.n_poi_types
        # How many of each POI type are in this room
        # Theoretically this part could be calculated only once with how it is set up right now
        # BUT I'm leaving it this way because we will likely have POIs that appear / disappear at some point
        for poi in self.pois:
            if poi.observed:
                continue
            st[poi.type] += 1
        # ag_st = [0] * self.p.n_agent_types
        # # How many of each agent type are in this room
        # for ag in self.agents_in_rm:
        #     ag_st[ag.type] += 1
        # st.extend(ag_st)
        return st