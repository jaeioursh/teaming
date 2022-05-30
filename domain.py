import numpy as np
import matplotlib.pyplot as plt

class POI:
    def __init__(self, x, y, value, refresh_rate, obs_required, couple, poi_type,n_agents, strong_coupling=True):
        self.value = value                  # POI value -- this only makes sense for some reward structures
        self.successes = 0                  # number of times it has successfully been captured
        self.D_vec=np.zeros(n_agents)
        self.refresh_idx = 0                # time steps since last refresh
        self.refresh_rate = refresh_rate    # how often it is refreshed
        self.obs_required = obs_required    # number of observations required to fully observe the POI
        self.obs_radius = 2                 # observation radius
        self.couple = couple                # coupling requirement
        self.x = x                          # location - x
        self.y = y                          # location - y
        self.poi_type = poi_type            # type
        self.strong_coupling = strong_coupling  # 1: simultaneous obvservation,  0: observations within window of time
        self.viewed = []                    # list of all agents that viewed in refresh window
        self.viewing = []                   # list of currently observing agents


    def reset(self):
        """
        Reset refresh, successes, viewed, and viewing
        :return:
        """
        self.refresh_idx = 0
        self.successes = 0
        self.D_vec[:]=0
        self.viewed = []
        self.viewing = []

    def refresh(self):
        self.refresh_idx += 1                               # increase number of time steps since last refresh
        if self.refresh_idx == self.refresh_rate:           # if it has hit the refresh
            self.refresh_idx = 0                            # reset the time steps
            if self.strong_coupling:                        # if it requires simultaneous observation
                if len(self.viewing) >= self.couple:        # if the number of simultaneous observations eets the coupling requirement
                    capabilities = [agent.capabilities[self.poi_type] for agent in self.viewing]    # check to make sure the agents are capable of observing this POI
                    capabilities=sorted(capabilities)
                    g=capabilities[0]     # Add minimum capability of agents to success of observation
                    self.successes += g
                    #difference reward block
                    idxs=[agent.idx for agent in self.viewing]
                    d=[]
                    for agent in self.viewing:
                        if len(self.viewing)>=self.couple+1:
                            if agent.capabilities[self.poi_type]==g:
                                d.append(g-capabilities[1])
                            else:
                                d.append(0)
                        else:
                            d.append(g)               
                    self.D_vec[idxs]+=np.array(d)
                    
            else:
                if len(self.viewed) >= self.couple:         # if weak coupling, check all the agents that viewed this refresh cycle
                    capabilities = [agent.capabilities[self.poi_type] for agent in self.viewed]
                    g=min(capabilities)
                    self.successes += g

            self.viewed = []                                # reset the agents that have viewed


class Agent:
    def __init__(self, x, y, N_pois,idx):
        self.idx=idx
        self.x = x                  # location - x
        self.y = y                  # location - y
        self._x = x                 # initial location - x
        self._y = y                 # initial location - y
        self.poi = None             # variable to store desired POI
        self.capabilities = np.random.random(N_pois)    # randomly initialize agent's capability of viewing each POI

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
            return 1
        else:
            return 0


class DiscreteRoverDomain:
    def __init__(self, N_agents, N_pois):
        self.N_agents = N_agents            # number of agents
        self.N_pois = N_pois                # number of POIs

        self.size = 30                      # size of the world

        self.agents = self.gen_agents()     # generate agents
        self.pois = self.gen_pois()         # generate POIs
        self.reset()                        # reset the system
        self.vis=0

    def draw(self):
        if self.vis==0:
            self.vis=1
            plt.ion()
            self.agent_offset=np.random.normal(0,0.5,(self.N_agents,2))
        plt.clf()
        xy=np.array([[poi.x,poi.y] for poi in self.pois])
        XY=np.array([[agent.x,agent.y] for agent in self.agents])+self.agent_offset
        alpha=[1-poi.refresh_idx/poi.refresh_rate for poi in self.pois]
        sizes=[poi.value*20+10 for poi in self.pois]
        print(alpha)
        plt.scatter(xy[:,0],xy[:,1],marker="s",s=sizes,c=alpha,vmin=0,vmax=1)
        plt.scatter(XY[:,0],XY[:,1],marker="o")
        plt.ylim([0,self.size])
        plt.xlim([0,self.size])
        plt.pause(0.1)
    # generate list of agents
    def gen_agents(self):
        """
        Generates a list of agents
        :return: list of agent objects at random initial locations
        """
        # creates an array of x, y positions for each agent
        # locations are [0,4] plus half the size of the world
        X = np.random.randint(0, 4, self.N_agents) + self.size // 2
        Y = np.random.randint(0, 4, self.N_agents) + self.size // 2
        idxs=[i for i in range(self.N_agents)]
        # return an array of Agent objects at the specified locations
        return [Agent(x, y, self.N_pois,i) for x, y, i in zip(X,Y,idxs)]

    # generate list of POIs
    def gen_pois(self):
        """
        Generates a list of random POIs
        :return: list of POI objects
        """
        x = np.random.randint(0, self.size, self.N_pois)        # x locations for all POIs
        y = np.random.randint(0, self.size, self.N_pois)        # y locations for all POIs
        # these need to be less hard-coded
        refresh_rate = [10 for i in range(self.N_pois)]         # refresh rate for all POIs
        obs_required = [2 for i in range(self.N_pois)]          # number of observations required
        couple = [2 for i in range(self.N_pois)]                # coupling requirement for all POIs
        poi_type = [i for i in range(self.N_pois)]              # Each one is a different type
        value = poi_type                                        # Value of the POI
        n_agents=[self.N_agents]*self.N_pois
        # return a list of the POI objects
        return list(map(POI, x, y, value, refresh_rate, obs_required, couple, poi_type,n_agents))

    # reset environment to intial config
    def reset(self):
        """
        Reset environment to initial configuration
        :return:
        """
        for a in self.agents:       # reset all agents to initial config
            a.reset()
        for p in self.pois:         # reset all POIs to initial config
            p.reset()

    # perform one state transition given a list of actions for each agent
    def step(self, actions):
        """
        perform one state transition given a list of actions for each agent
        :param actions:
        :return:
        """
        # refresh all POIs and reset which agents are currently viewing
        for i in range(self.N_pois):
            self.pois[i].refresh()
            self.pois[i].viewing = []                   # if this gets reset at every step, the "viewing" check will only see the last time step

        # update all agents
        for i in range(self.N_agents):
            self.agents[i].poi = self.pois[actions[i]]  # agents set a new goal at every time step
            self.agents[i].move()                       # move agent toward POI
            if self.agents[i].observe():                # If at the POI and observed
                poi = self.agents[i].poi                # get the POI ID
                poi.viewing.append(self.agents[i])      # add the agent to current agents viewing the POI
                if self.agents[i] not in poi.viewed:    # this logic assumes each POI needs to be visited by different agents
                    poi.viewed.append(self.agents[i])

    # TODO
    def state(self):
        pass

    # returns global reward based on POI values
    def G(self):
        g = 0
        for poi in self.pois:
            g += poi.successes * poi.value
        return g

    def D(self):
        d=np.zeros(self.N_agents)
        for poi in self.pois:
            d = d +  poi.D_vec * poi.value
        return d

if __name__ == "__main__":
    np.random.seed(0)
    env = DiscreteRoverDomain(3, 6)
    for i in range(100):
        actions = [3, 3, 0]
        env.step(actions)
        env.draw()
        print(i, env.G(),env.D())
