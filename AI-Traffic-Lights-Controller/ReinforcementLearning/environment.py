from typing import Optional, List, Tuple

from TrafficSimulator import Simulation
from TrafficSimulator.Setups import two_way_intersection_setup
from TrafficSimulator.vehicle import BRT


class Environment:
    def __init__(self):
        self.action_space: List = [0, 1]
        self.sim: Optional[Simulation] = None
        self.max_gen: int = 50
        self._vehicles_on_inbound_roads: int = 0

    def step(self, step_action) -> Tuple[Tuple, float, bool, bool]:
        self.sim.run(step_action)

        new_state: Tuple = self.get_state()

        step_reward: float = self.get_reward(new_state)

        # Set the number of vehicles on inbound roads in the new state
        n_west_east_vehicles, n_south_north_vehicles = new_state[1], new_state[2]
        self._vehicles_on_inbound_roads = n_west_east_vehicles + n_south_north_vehicles

        # Whether a terminal state (as defined under the MDP of the task) is reached.
        terminated: bool = self.sim.completed

        # Whether a truncation condition outside the scope of the MDP is satisfied.
        # Ends the episode prematurely before a terminal state is reached.
        truncated: bool = self.sim.gui_closed

        return new_state, step_reward, terminated, truncated

    def get_state(self) -> Tuple:
        """ A state is a tuple, with sub-tuples representing junctions with traffic signals.
        Each sub-tuple is contains the following stats: the traffic signal state, the number
        of vehicles in the 1st direction, the number of vehicles in the 2nd direction,
        and an indicator of whether the junction is empty or not """
        state = []
        for traffic_signal in self.sim.traffic_signals:
            junction = []
            traffic_signal_state = traffic_signal.current_cycle[0]
            junction.append(traffic_signal_state)

            for direction in traffic_signal.roads:
                junction.append(sum(len(road.vehicles) for road in direction))

            n_direction_1_vehicles, n_direction_2_vehicles = junction[1], junction[2]
            out_bound_vehicles = sum(len(self.sim.roads[i].vehicles) for i in self.sim.outbound_roads)
            non_empty_junction = bool(self.sim.n_vehicles_on_map - out_bound_vehicles -
                                      n_direction_1_vehicles - n_direction_2_vehicles)
            junction.append(non_empty_junction)
            state.append(junction)
        state = state[0]  # Optimization for a single junction simulation setup
        return tuple(state)

    def get_reward(self, state: Tuple) -> float:
        """ Check the flow change and waiting time of BRT vehicles.
        Give higher rewards for allowing BRT vehicles to pass quickly."""
        traffic_signal_state, n_direction_1_vehicles, n_direction_2_vehicles, non_empty_junction = state
        
        # Get the flow change
        flow_change = self._vehicles_on_inbound_roads - n_direction_1_vehicles - n_direction_2_vehicles
        
        # Get the waiting time of BRT vehicles
        brt_wait_time = 0
        for road_index in self.sim.inbound_roads:
            road = self.sim.roads[road_index]
            for vehicle in road.vehicles:
                if isinstance(vehicle,BRT):
                    brt_wait_time += vehicle._waiting_time
        
        # Combine the flow change, BRT waiting time, and BRT passed with weight factors
        # print("flowchnage : ", flow_change, "brt_wait_time : ", brt_wait_time)
        reward = flow_change + 1.8 * flow_change - 0.6 * brt_wait_time
        # print(reward)
        return reward


    def reset(self, render=False) -> Tuple:
        self.sim = two_way_intersection_setup(self.max_gen)
        if render:
            self.sim.init_gui()
        init_state = self.get_state()
        self._vehicles_on_inbound_roads = 0  # Reset the counter
        return init_state
