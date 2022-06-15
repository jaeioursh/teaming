import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from teaming.POI import POI
from teaming.Agent import Agent

class DiscreteRoverDomain:
    def __init__(self, N_agents, N_pois=4):
        self.N_agents = N_agents            # number of agents
        self.N_pois = N_pois                # number of POIs

        self.size = 30                      # size of the world

        self.pois = self.gen_pois()         # generate POIs
        self.agents = self.gen_agents()     # generate agents
        
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
        return [Agent(x, y, np.random.random(self.N_pois),i) for x, y, i in zip(X,Y,idxs)]

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
        
        # update all agents
        for i in range(self.N_agents):
            if actions[i]==self.N_pois:
                continue
            if actions[i] is not None:
                self.agents[i].poi = self.pois[actions[i]]  # agents set a new goal at every time step
            self.agents[i].move()                       # move agent toward POI
            if self.agents[i].observe():                # If at the POI and observed
                poi = self.agents[i].poi                # get the POI ID
                poi.viewing.append(self.agents[i])      # add the agent to current agents viewing the POI
                if self.agents[i] not in poi.viewed:    # this logic assumes each POI needs to be visited by different agents
                    poi.viewed.append(self.agents[i])

        # refresh all POIs and reset which agents are currently viewing
        for i in range(self.N_pois):
            self.pois[i].refresh()
            self.pois[i].viewing = []                   # if this gets reset at every step, the "viewing" check will only see the last time step

    
    def state(self):
        s_poi=[float(not poi.refresh_idx) for poi in self.pois]
        S=[]
        for agent in self.agents:
            
            """
            s=[]
            for poi in self.pois:
                s.append(( abs(agent.x-poi.x)+abs(agent.y-poi.y) ) / self.size )
            """
            s=[agent.x/self.size,agent.y/self.size]
            S.append(s_poi+s)
            

        return np.array(S)
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
    def Dpp(self):
        d=np.zeros(self.N_agents)
        for poi in self.pois:
            d = d +  poi.Dpp_vec * poi.value
        return d

if __name__ == "__main__":
    np.random.seed(0)
    env = DiscreteRoverDomain(3, 6)
    for i in range(30):
        actions = [3, 3, 3]
        env.step(actions)
        
        env.draw()
        print(i, env.G(),env.D())
    print(env.state())
 