import numpy as np

from teaming.domain import DiscreteRoverDomain as Domain


class DomainHierarchy(Domain):
    def __init__(self, p, reselect):
        super().__init__(p)
        self.reselect_ts = reselect  # How often they re-select a policy
        self.pareto_vals = None
        self.pareto_pols = None
        self.policies = None
        self.max_gs = None

    def setup(self, pareto_vals, pareto_pols):
        # Have to do this separately because the top level CCEA requires some info from domain to set up
        # But also this requires info from top level CCEA to set up
        # This avoids a catch-22 of information
        self.pareto_vals = pareto_vals
        self.pareto_pols = pareto_pols
        self.max_gs = self.find_max_gs()

    def find_max_gs(self):
        # Max G0 for each agent
        max_vals = []
        for i in range(self.n_agents):
            max_vals.append(self.pareto_vals[i][-1][0])
        return max_vals

    def run_sim(self, top_policies):
        for t in range(self.time_steps):
            if not t % self.reselect_ts:
                self.reselect_policies(top_policies)

            for i in range(len(self.policies)):
                self.agents[i].policy = self.policies[i]  # sets all agent policies

            self.time = t
            state = self.joint_state()
            actions = self.joint_actions(state)
            self.step(actions)
            if self.vis:
                self.view(t)

        return self.G()

    def reselect_policies(self, top_pols):
        self.policies = []
        st = self.global_state()
        for i, pol in enumerate(top_pols):
            # TODO: Double check that this is correct
            poi_scale = pol(st).detach().numpy()[0] * self.max_gs[i]
            idx = closest(poi_scale, self.pareto_vals[i])
            self.policies.append(self.pareto_pols[i][idx])

    def global_state(self):
        # st = np.zeros(self.n_poi_types + self.n_agent_types)
        st = np.zeros(self.n_poi_types)

        # How many of each poi type have been captured
        for poi in self.pois:
            if poi.observed:
                st[poi.type] += 1

        # How many of each agent type
        # for ag in self.agents:
        #     st[self.n_poi_types + ag.type] += 1

        return st

    def global_st_size(self):
        # return self.n_poi_types + self.n_agent_types
        return self.n_poi_types

    def high_level_G(self):
        g = self.multiG()
        possible_G = np.sum(self.p.rooms, axis=0)
        if g[0] < int(possible_G[0]/2):
            return 0
        elif g[1] > 0:
            return 0
        else:
            return 1


def closest(num, arr):
    curr = arr[0][0]
    idx = 0
    for i, val in enumerate(arr):
        if abs(num - val[0]) < abs(num - curr):
            curr = val[0]
            idx = i
    return idx