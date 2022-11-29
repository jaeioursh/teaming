import numpy as np

class Dummy:
    def __init__(self):
        self.size = 30
        self.rooms = [[1,2,3], [3,2,1], [4,5,6], [2,1,4], [2,5,1]]

    def gen_world(self):
        room_dims = []
        # Hallway x, y positions
        hall_st = np.floor(self.size / 2)
        hall_lower = [0, hall_st]  # x, y of hallway lower left corner
        hall_upper = [self.size, hall_st + 1]  # x, y of hallway upper right corner

        # Decide how many rooms are above / below the hallway
        n_rooms = int(len(self.rooms))
        n_upper = int(np.floor(n_rooms / 2))
        n_lower = n_upper
        if n_rooms % n_upper:
            # If there are leftovers after splitting into two groups, add one to the lower group
            n_lower = n_upper + 1

        # x axis dimensions
        upper_bins = np.linspace(0, self.size, n_upper + 1)
        lower_bins = np.linspace(0, self.size, n_lower + 1)
        upper_x = []
        lower_x = []
        for i in range(len(upper_bins) - 1):
            upper_x.append([upper_bins[i], upper_bins[i + 1]])
        for j in range(len(lower_bins) - 1):
            lower_x.append([lower_bins[j], lower_bins[j + 1]])

        # y axis dimensions
        upper_y = [[hall_st + 1, self.size] for _ in range(n_upper)]
        lower_y = [[0, hall_st] for _ in range(n_lower)]
        upper_dims = np.dstack((upper_x, upper_y))
        lower_dims = np.dstack((lower_x, lower_y))

        # Note to future self: you now have [[x0, y0], [x1, y1]] for the halls and rooms
        # This has been tested with 2 and 5 rooms and works for both values.
        # Do with that what you will. But also you're welcome.

        pass


if __name__ == '__main__':
    dummy = Dummy()
    dummy.gen_world()
