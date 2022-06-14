import numpy as np
from teaming.visuals_test import VisualizeDomain as Visualize
from teaming.POI import POI
from teaming.agent import Agent
from qlearner.qlearner.qlearner import QLearner
from time import time


class DiscreteRoverDomain:
    def __init__(self, N_agents, N_pois, poi_options):
        self.N_agents = N_agents            # number of agents
        self.N_pois = N_pois                # number of POIs
        self.poi_options = np.array(poi_options)      # options for each POI - [refresh rate, number of observations required, ID]
        self.num_poi_options = np.shape(self.poi_options)[0]    # how many different types of POIs - allows for calc of L
        self.size = 30                      # size of the world
        self.time_steps = 100               # time steps per epoch
        self.agents = self.gen_agents()     # generate agents
        self.pois = self.gen_pois()         # generate POIs
        self.viz = Visualize(self.size, self.pois, self.agents)
        self.reset()                        # reset the system

    # generate list of agents
    def gen_agents(self):
        """
        Generates a list of agents
        :return: list of agent objects at random initial locations
        """
        # creates an array of x, y positions for each agent
        # locations are [0,4] plus half the size of the world
        self.starting_locs = np.random.randint(0, 4, (self.N_agents, 2)) + self.size // 2
        # return an array of Agent objects at the specified locations
        return [Agent(x, y, self.N_pois) for x, y in self.starting_locs]

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

        # refresh_rate = [10 for i in range(self.N_pois)]         # refresh rate for all POIs
        # obs_required = [2 for i in range(self.N_pois)]          # number of observations required
        # poi_type = [i for i in range(self.N_pois)]              # Each one is a different type

        refresh_rate = np.ndarray.tolist(poi_vals[:, 0])
        obs_required = np.ndarray.tolist(poi_vals[:, 1])
        poi_type = np.ndarray.tolist(poi_vals[:, 2])

        couple = [2 for _ in range(self.N_pois)]                # coupling requirement for all POIs
        value = [1 for _ in range(self.N_pois)]                 # Value of the POI
        poi_idx = [i for i in range(self.N_pois)]
        # return a list of the POI objects
        return list(map(POI, x, y, value, refresh_rate, obs_required, couple, poi_type, poi_idx))

    # reset environment to intial config
    def reset(self):
        """
        Reset environment to initial configuration
        :return:
        """
        for a in self.agents:       # reset all agents to initial config
            a.reset()
            a.curr_l = np.zeros(self.num_poi_options)
            a.tot_l = np.zeros(self.num_poi_options)
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
            # if i == 0:
            #     print(self.state(self.agents[i]))
            if actions[i] >= self.N_pois:
                # The policy has chosen the "null" action
                continue
            self.agents[i].step(self.pois[actions[i]])

        # refresh all POIs and reset which agents are currently viewing
        for j in range(self.N_pois):
            self.pois[j].refresh()
        for k in range(self.N_agents):
            self.L(self.agents[k])
        # self.viz.show_grid()

    def state(self, agent):
        ax, ay = agent.x, agent.y
        distance = np.zeros(self.N_pois)
        for poi in self.pois:
            px, py = poi.x, poi.y
            d = abs(ax - px) + abs(ay - py)     # find the square distance between the agent and each POI
            distance[poi.poi_idx] = d
            # if d < 10:
            #     distance[poi.poi_idx] = 0
            # else:
            #     distance[poi.poi_idx] = 1
        # last_visit = np.zeros_like(agent.last_visit)
        # for i in range(len(last_visit)):
        #     if agent.last_visit[i] < 10:
        #         last_visit[i] = 0
        #     else:
        #         last_visit[i] = 1
        state = np.concatenate((distance, agent.last_visit))
        # stupid_state = ""
        # for i in state:
        #     stupid_state += str(int(i))
        return state

    def state_size(self):
        return self.N_pois * 2

    # returns global reward based on POI values
    def G(self):
        g = 0
        for poi in self.pois:
            g += poi.successes * poi.value
        return g

    def L(self, agent):
        rew_arr = np.zeros(self.num_poi_options)        # each POI type associates with a different L
        for poi in self.pois:
            if poi.history and poi.curr_rew:
                if agent in poi.history[-1]:
                    rew_arr[poi.poi_type] += 1
        agent.curr_l = rew_arr
        agent.tot_l += rew_arr
        # print(agent.curr_l, agent.tot_l)

    def run_sim(self, policies):
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
                actions.append(act_detensorfied)             # save the action to list of actions
            self.step(actions)

        return self.G()


if __name__ == "__main__":
    start = time()
    np.random.seed(1)
    # POI types should be of the form [refresh rate, number of observations required, unique ID]
    poi_types = [[100, 1, 0]]  #, [10, 3, 1], [50, 5, 2]]
    num_agents = 2
    num_pois = 30
    num_states = 2**(num_pois*2)

    for blerg in range(1000):
        env = DiscreteRoverDomain(num_agents, num_pois, poi_types)
        env.reset()
        # q_learners = []
        # for _ in range(num_agents):
        #     q_learners.append(QLearner(num_states, num_pois))
        #
        #
        # state = np.zeros(num_agents).astype(int)
        # prev_state = np.zeros(num_agents).astype(int)
        # for i in range(num_agents):
        #     st = env.state(env.agents[i])
        #     state[i] = st

        for l in range(100):
            # print("################# {} ###############".format(l))
            # actions = []
            actions = np.ndarray.tolist(np.random.randint(3, size=num_agents))
            # for i in range(num_agents):
            #     act = q_learners[i].selectAction(state[i])
            #     actions.append(act)

            env.step(actions)
            # prev_state = state

            # for i in range(num_agents):
            #     st = env.state(env.agents[i])
            #     state[i] = st
            #     L = env.agents[i].curr_l
            #     q_learners[i].updateQValue(prev_state[i], actions[i], state[i], sum(L))

        print(blerg, env.G())

    print(time() - start)
