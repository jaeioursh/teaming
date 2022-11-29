"""
Example parameters file for the rover domain.
"""


class Parameters:
    # This should match the file name -- parameters##
    trial_num = 00

    # Domain:
    n_agents = 1
    n_agent_types = 1
    n_poi_types = 3
    rooms = [[1, 2, 3], [3, 2, 1], [3, 3, 0], [3, 0, 3]]
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
