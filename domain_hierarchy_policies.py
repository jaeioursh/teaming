import numpy as np

from scipy.spatial.distance import cdist
from teaming.domain import DiscreteRoverDomain as Domain


class DomainHierarchy(Domain):
    def __init__(self, p, reselect):
        super().__init__(p)
        self.reselect_ts = reselect  # How often they re-select a policy
        self.g_vals = None
        self.pareto_pols = None
        self.behaviors = None
        self.policies = None
        self.norm_gs_bh = []

    def setup(self, pareto_vals, pareto_pols, behaviors):
        # Have to do this separately because the top level CCEA requires some info from domain to set up
        # But also this requires info from top level CCEA to set up
        # This avoids a catch-22 of information
        self.g_vals = pareto_vals
        self.pareto_pols = pareto_pols
        self.behaviors = behaviors
        for i, gs in enumerate(self.g_vals):
            norm_gs = gs / gs.max(axis=0)
            g_and_bh = np.concatenate((norm_gs, behaviors[i]), axis=1)
            self.norm_gs_bh.append(g_and_bh)

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
            nn_out = pol(st).detach().numpy()
            idx = closest(nn_out, self.norm_gs_bh[i])
            self.policies.append(self.pareto_pols[i][idx])

    def global_state(self):
        st = np.zeros(self.n_poi_types + self.n_agent_types)
        # st = np.zeros(self.n_poi_types)

        # How many of each poi type have been captured
        for poi in self.pois:
            if poi.observed:
                st[poi.type] += 1

        # How many of each agent type
        for ag in self.agents:
            st[self.n_poi_types + ag.type] += 1

        return st

    def global_st_size(self):
        return self.n_poi_types + self.n_agent_types
        # return self.n_poi_types

    def top_out_size(self):
        # Output size for top-level policy
        # Number of POI types is currently the number of objectives we're balancing
        # Number of rooms is the number of behaviors
        return self.n_poi_types + self.n_rooms

    def high_level_G(self):
        g = self.multiG()
        possible_G = np.sum(self.p.rooms, axis=0)
        if g[0] < int(possible_G[0]/2):
            return 0
        elif g[1] > 0:
            return 0
        else:
            return 1


def closest(nn_out, arr):
    # idx = (np.abs(arr - nn_out)).sum(axis=1).argmin()
    subtract_them = np.abs(arr - nn_out)
    sum_them = subtract_them.sum(axis=1)
    find_min = sum_them.argmin()
    return find_min