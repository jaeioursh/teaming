import numpy as np
from math import pi, sqrt, atan2
import matplotlib.pyplot as plt

from teaming.POI import POI
from teaming.Agent import Agent


class DiscreteRoverDomain:
    def __init__(self, N_agents, N_pois=4, poi_options=[[100, 1, 0]]):
        self.N_agents = N_agents            # number of agents
        self.N_pois = N_pois                # number of POIs
        self.poi_options = np.array(poi_options)      # options for each POI - [refresh rate, number of observations required, ID]
        self.num_poi_options = np.shape(self.poi_options)[0]    # how many different types of POIs - allows for calc of L
        self.size = 30                      # size of the world
        self.time_steps = 100               # time steps per epoch
        self.agents = self.gen_agents()     # generate agents
        self.pois = self.gen_pois()         # generate POIs
        self.n_bins = 8                     # number of bins (quadrants) for sensor discretization
        self.sensor_bins = np.linspace(pi, -pi, self.n_bins+1, True)    # Discretization for sensor bins
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
        num_poi_types = np.shape(self.poi_options)[0]
        randints = np.random.randint(num_poi_types, size=self.N_pois)     # get an array of random integers - integers represent the POI type, one for each POI in the world
        poi_vals = self.poi_options[randints, :]                     # array of values from the POI options for each POI in the world
        x = np.random.randint(0, self.size, self.N_pois)        # x locations for all POIs
        y = np.random.randint(0, self.size, self.N_pois)        # y locations for all POIs
        # Each one is a different type
        n_agents=[self.N_agents]*self.N_pois

        # these need to be less hard-coded
        # refresh_rate = [10 for i in range(self.N_pois)]         # refresh rate for all POIs
        # obs_required = [2 for i in range(self.N_pois)]          # number of observations required
        # poi_type = [i for i in range(self.N_pois)]              # Each one is a different type
        # value = poi_type                                        # Value of the POI

        refresh_rate = np.ndarray.tolist(poi_vals[:, 0])
        obs_required = np.ndarray.tolist(poi_vals[:, 1])
        poi_type = np.ndarray.tolist(poi_vals[:, 2])

        couple = [2 for _ in range(self.N_pois)]                # coupling requirement for all POIs
        value = [1 for _ in range(self.N_pois)]                 # Value of the POI
        poi_idx = [i for i in range(self.N_pois)]
        # return a list of the POI objects
        return list(map(POI, x, y, value, refresh_rate, obs_required, couple, poi_type, n_agents, poi_idx))

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
            if actions[i] >= self.N_pois:
                # The policy has chosen the "null" action
                continue
            if actions[i] is not None:
                self.agents[i].poi = self.pois[actions[i]]  # agents set a new goal at every time step
            # TODO: Check what of the below is in the Agent move class
            # self.agents[i].move()  # move agent toward POI
            # if self.agents[i].observe():  # If at the POI and observed
            #     poi = self.agents[i].poi  # get the POI ID
            #     poi.viewing.append(self.agents[i])  # add the agent to current agents viewing the POI
            #     if self.agents[
            #         i] not in poi.viewed:  # this logic assumes each POI needs to be visited by different agents
            #         poi.viewed.append(self.agents[i])

        # refresh all POIs and reset which agents are currently viewing
        for i in range(self.N_pois):
            self.pois[i].refresh()
            self.pois[i].viewing = []                   # if this gets reset at every step, the "viewing" check will only see the last time step

    #TODO: Rewrite with current state space design
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

    def _get_quadrant_state_info(self, agent, a_or_p='p'):
        """
        Get 'quadrant' state information for all other points or agents
        Parameters
        ----------
        agent:  agent for which we are getting state info
        a_or_p: is the state info for all other agents or POIs

        Returns
        -------
        distance and quadrant number for each point

        """
        # tested this and it should be accurate
        if a_or_p == 'p':
            num_points = self.N_pois
            points = self.pois
        else:
            num_points = self.N_agents
            points = self.agents
        dist_arr = np.zeros(num_points)        # distance to each POI
        theta_arr = np.zeros(num_points)       # angle to each poi [-pi, pi]
        for i in range(num_points):
            point = points[i]
            x = point.x - agent.x
            y = point.y - agent.y
            dist_arr[i] = sqrt(x**2 + y**2)     # absolute distance to each POI
            theta_arr[i] = atan2(y, x)          # angle to each POI
        quadrants = np.digitize(theta_arr, bins=self.sensor_bins)-1     # which quadrant / octant each POI is in relative to the GLOBAL frame
        return dist_arr, quadrants

    #TODO: Rewrite with current state space
    def state_size(self):
        return self.N_pois * 2

    # returns global reward based on POI values
    def G(self):
        g = 0
        for poi in self.pois:
            g += poi.successes * poi.value
        return g

    def D(self):
        d=np.zeros(self.N_agents)
        for poi in self.pois:
            d = d + poi.D_vec * poi.value
        return d

    def Dpp(self):
        d=np.zeros(self.N_agents)
        for poi in self.pois:
            d = d + poi.Dpp_vec * poi.value
        return d

    def run_sim(self, policies):
        """
        This is set up to test a set of NN policies, one for each agent.
        Parameters
        ----------
        policies: a policy (assumes a NN) for each agent

        Returns
        -------
        G: global reward
        """

        if len(policies) != self.N_agents:
            raise ValueError('number of policies should equal number of agents in system (currently {})'.format(self.N_agents))

        for i in range(len(policies)):
            self.agents[i].policy = policies[i]

        for _ in range(self.time_steps):
            actions = []
            for agent in self.agents:
                st = self.state(agent)          # gets the state
                act = agent.policy(st)          # picks an action based on the policy
                act_detensorfied = np.argmax(act.detach().numpy())  # converts tensor to numpy, then finds the index of the max value
                #TODO: add check to make sure action is valid
                actions.append(act_detensorfied)             # save the action to list of actions
            self.step(actions)

        return self.G()


if __name__ == "__main__":
    np.random.seed(0)
    poi_types = [[100, 1, 0]]  #, [10, 3, 1], [50, 5, 2]]
    num_agents = 2
    num_pois = 30
    num_states = 2**(num_pois*2)
    env = DiscreteRoverDomain(num_agents, num_pois, poi_types)

    for i in range(30):
        actions = [3, 3, 3]
        env.step(actions)

        env.draw()
        print(i, env.G(), env.D())
    print(env.state())
