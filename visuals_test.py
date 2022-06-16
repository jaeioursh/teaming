from matplotlib import pyplot as plt
import numpy as np
from matplotlib import colors
import time
from pylab import cm

class VisualizeDomain:
    def __init__(self, size, pois, agents):
        self.size = size
        self.pois = pois
        self.agents = agents
        self.grid_world = np.zeros((self.size, self.size))
        self.set_grid()
        self.step_num = 0
        # self.show_grid()

    def set_grid(self):
        self.grid_world = np.zeros((self.size, self.size))
        for poi in self.pois:
            x, y = poi.x, poi.y
            self.grid_world[x][y] += 1
        # print(self.grid_world)

    def show_grid(self):
        # cmap = plt.cm.rainbow
        # norm = colors.BoundaryNorm(np.arange(0.1, 3, 1), cmap.N)
        # cmap.set_under(color='white')
        plt.clf()
        plt.figure(figsize=(8, 8))
        for p in self.pois:
            plt.plot(p.x+0.5, p.y+0.5, marker='s', markersize=15, markerfacecolor='blue')
        for a in self.agents:
            plt.plot(a.x+0.5, a.y+0.5, marker='o', markersize=10, markerfacecolor='green')
        plt.xlim(0, 30)
        plt.ylim(0, 30)
        plt.savefig('images/dummy_fig{}'.format(self.step_num))
        self.step_num += 1

class fakeAgent:
    def __init__(self):
        self.x = np.random.randint(0, 30)
        self.y = np.random.randint(0, 30)

class fakePOI:
    def __init__(self):
        self.x = np.random.randint(0, 30)
        self.y = np.random.randint(0, 30)


if __name__ == '__main__':
    n_poi = 5
    n_agent = 3
    pois = []
    agents = []
    viz = VisualizeDomain(30, pois, agents)
    for _ in range(1):
        pois = []
        agents = []
        for _ in range(n_poi):
            dummy = fakePOI()
            pois.append(dummy)
        for _ in range(n_agent):
            dummy = fakeAgent()
            agents.append(dummy)
        viz.pois = pois
        viz.agents = agents
        # viz.set_grid()
        viz.show_grid()


# plt.grid(visible=None, which='major', axis='both')
# plt.show()