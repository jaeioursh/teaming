import numpy as np

class Room:
    def __init__(self, idx, p, bounds, door, pois):
        self.idx = idx
        self.p = p
        self.bounds = bounds  # [lower left, upper right] - [[x1, y1], [x2, y2]]
        self.pois = pois
        self.door = door
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
            st[poi.poi_type] += 1
        ag_st = [0] * self.p.n_agent_types
        # How many of each agent type are in this room
        for ag in self.agents_in_rm:
            ag_st[ag.type] += 1
        st.extend(ag_st)
        return st