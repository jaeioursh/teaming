
import numpy as np

class POI:
    def __init__(self,x,y,value,refresh_rate,couple,poi_type,strong_coupling=True):
        self.value=value
        self.successes=0
        self.refresh_idx=0
        self.obs_radius=2
        self.couple=couple
        self.refresh_rate=refresh_rate
        self.x=x
        self.y=y
        self.poi_type=poi_type
        self.strong_coupling=strong_coupling
        self.viewed=[]      #list of currently observing agents
        self.viewing=[]     #list of all agents that viewed in refresh window
        self.history=[]

    def reset(self):
        self.refresh_idx=0
        self.successes=0
        self.viewed=[]
        self.viewing=[]
    
    def refresh(self):
        self.refresh_idx+=1
        if self.refresh_idx==self.refresh_rate:
            self.refresh_idx=0
            if self.strong_coupling:
                if len(self.viewing)>=self.couple:
                    capabilities=[agent.capabilities[self.poi_type] for agent in self.viewing]
                    self.successes+=min(capabilities)
                    self.history.append(self.viewing)
            else:
                if len(self.viewed)>=self.couple:
                    capabilities=[agent.capabilities[self.poi_type] for agent in self.viewed]
                    self.successes+=min(capabilities)
                    self.history.append(self.viewed)
            self.viewed=[]
        

class Agent:
    def __init__(self,x,y,N_pois):
        self.x=x
        self.y=y
        self._x=x
        self._y=y

        self.poi=None
        self.capabilities=np.random.random(N_pois)

    def reset(self):
        self.x=self._x
        self.y=self._y
        self.poi=None
        

    def move(self):
        if self.poi is not None:
            X=self.poi.x
            Y=self.poi.y
            if X>self.x:
                self.x+=1
            elif X<self.x:
                self.x-=1
            elif Y>self.y:
                self.y+=1
            elif Y<self.y:
                self.y-=1
    def observe(self):
        if abs(self.poi.x-self.x)<self.poi.couple and abs(self.poi.y-self.y)<self.poi.couple:
            return 1
        else:
            return 0


class DiscreteRoverDomain:
    def __init__(self,N_agents,N_pois):
        self.N_agents=N_agents
        self.N_pois=N_pois
        
        self.size=30

        self.agents=self.gen_agents()
        self.pois=self.gen_pois()
        self.reset()

    def gen_agents(self):
        self.starting_locs=np.random.randint(0,4,(2,self.N_agents))+self.size//2
        return [Agent(x,y,self.N_pois) for x,y in self.starting_locs]

    def gen_pois(self):
        x=np.random.randint(0,self.size,(self.N_pois))
        y=np.random.randint(0,self.size,(self.N_pois))
        refresh_rate=[10 for i in range(self.N_pois)]
        couple=[2 for i in range(self.N_pois)]
        poi_type=[i for i in range(self.N_pois)]
        value=poi_type
        return list(map(POI,x,y,value,refresh_rate,couple,poi_type))

    def reset(self):
        for a in self.agents:
            a.reset()
        for p in self.pois:
            p.reset()

    def step(self,actions):
        for i in range(self.N_pois):
            self.pois[i].refresh()
            self.pois[i].viewing=[]
            
        for i in range(self.N_agents):
            self.agents[i].poi=self.pois[actions[i]]    
            self.agents[i].move()
            if self.agents[i].observe():
                poi=self.agents[i].poi
                poi.viewing.append(self.agents[i])
                if self.agents[i] not in poi.viewed:
                    poi.viewed.append(self.agents[i])
            
    def state(self):
        pass
    
    def G(self):
        g=0
        for poi in self.pois:
            g+=poi.successes*poi.value
        return g




if __name__=="__main__":
    np.random.seed(0)
    env=DiscreteRoverDomain(2,6)
    for i in range(100):
        
        actions=[3,3]
        env.step(actions)
        print(i,env.G())
