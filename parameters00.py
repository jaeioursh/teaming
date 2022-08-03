"""
Example parameters file for the rover domain.
"""


class Parameters:
    # This should match the file name -- parameters##
    trial_num = 00

    # Domain:
    n_agents = 1
    n_agent_types = 1
    n_pois = 10
    poi_options = [[60, 1, 1, 1], [30, 2, 1, 1]]  # time active, number of times active, observation_req, value
    with_agents = True
    size = 30
    time_steps = 100
    n_regions = 8
    sensor_range = 10
    rand_action_rate = 0.05

    # POI:
    value = 1
    obs_radius = 1
    couple = 1
    strong_coupling = False

    # Agent:
    capabilities = False
