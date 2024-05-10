from typing import List, Tuple

import numpy as np


class Vehicle:
    def __init__(self, path: List[int], priority: int = 2):
        self.index = 0
        self.length = 4
        self.width = 2

        self.s0 = 4
        self.T = 1
        self.v_max = 16.6  # Max velocity
        self.a_max = 1.44  # Max positive acceleration
        self.b_max = 4.61  # Max negative acceleration
        self.sqrt_ab = 2 * np.sqrt(self.a_max * self.b_max)
        self._v_max = self.v_max

        self.v = self.v_max  # Velocity
        self.a = 0  # Acceleration
        self.x = 0  # Position, relative to its current roadM

        self.is_stopped = False
        self._last_time_stopped = None
        self._waiting_time = 0

        self.path: List[int] = path  # Road indexes
        self.current_road_index = 0

        # Used for collision detection, value set upon adding it to the map in vehicle.update()
        self.position: Tuple = (None, None)

        # Set Priority
        self.priority = priority

        # Set Color
        self.color = (0, 0, 255)

    def __str__(self):
        return f'Vehicle {self.index}'

    def get_wait_time(self, sim_t):
        if self.is_stopped:
            return self._waiting_time + (sim_t - self._last_time_stopped)
        return self._waiting_time

    def update(self, lead, dt, road):
        """
        Updates the vehicle position velocity, and acceleration
        :param road: vehicle's road
        :param lead: vehicle
        :param dt: simulation time step
        """
        # Update position and velocity
        if self.v + self.a * dt < 0:
            self.x -= 1 / 2 * self.v * self.v / self.a
            self.v = 0
        else:
            self.v += self.a * dt
            self.x += self.v * dt + self.a * dt * dt / 2

        # Update acceleration
        alpha = 0
        if lead:
            delta_x = lead.x - self.x - lead.length
            delta_v = self.v - lead.v

            alpha = (self.s0 + max(0, self.T * self.v + delta_v * self.v / self.sqrt_ab)) / delta_x

        self.a = self.a_max * (1 - (self.v / self.v_max) ** 4 - alpha ** 2)

        if self.is_stopped:
            self.a = -self.b_max * self.v / self.v_max

        # Update position
        sin, cos = road.angle_sin, road.angle_cos
        x = road.start[0] + cos * self.x
        y = road.start[1] + sin * self.x
        self.position = x, y

    def stop(self, t):
        if not self.is_stopped:
            self._last_time_stopped = t
            self.is_stopped = True

    def unstop(self, t):
        if self.is_stopped:
            self._waiting_time += (t - self._last_time_stopped)
            self._last_time_stopped = None
            self.is_stopped = False

    def slow(self, traffic_light_slow_factor):
        self.v_max = self._v_max * traffic_light_slow_factor

    def unslow(self):
        self.v_max = self._v_max

class BRT(Vehicle):
    def __init__(self, path: List[int], priority: int = 1):
        super().__init__(path)
        self.priority = priority  # 우선순위 레벨 (1: 가장 높음)
        self.max_wait_time = 60  # 최대 대기 시간 (초)
        self._waited_time = 0  # 대기한 시간 누적
        self.length = 8
        self.color = (255, 0, 0)

    def update(self, lead, dt, road):
        super().update(lead, dt, road)
        
        # 우선순위에 따른 가속도 조정
        if self.priority == 1:
            self.a *= 1.2  
        else:
            self.a *= 1.0  # 20% 더 가속

            
    def __str__(self):
        return f'BRT {self.index} (Priority: {self.priority})'