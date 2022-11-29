import numpy as np

class Room:
    def __init__(self, p, bounds, door, pois):
        self.p = p
        self.bounds = bounds  # [lower left, upper right] - [[x1, y1], [x2, y2]]
        self.pois = pois
        self.door = door

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
        st = np.zeros(len(self.p.poi_options))
        for poi in self.pois:
            st[poi.poi_type] += 1
        return st