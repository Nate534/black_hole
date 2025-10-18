import numpy as np
from .constants import G, C

class BlackHole:

    def __init__(self, mass, position = (0,0,0)):
        self.position = np.array(position,dtype=np.float32)
        self.mass = mass 
        self.schwarz_radius = 2 * G * mass / (C ** 2)

    def get_rad_vec(self, position):
        return self.position - position
    
    def get_rad(self, r_vec):
        r_vec = np.array(r_vec, dtype=np.float64)
        relative_pos = r_vec - self.position

        return np.linalg.norm(relative_pos)

    def calc_grav_accel(self, position):
        position = np.array(position, dtype=np.float64)

        r_vec = self.get_rad_vec(position)
        r = self.get_rad(r_vec)

        if r == 0:
            return np.array([0.0, 0.0, 0.0])
        
        accel = G * self.mass / (r ** 2)
        direction = r_vec / r

        return accel * direction
    
    def is_inside_horizon(self, position):
        r = self.get_rad(position)
        return r <= self.schwarz_radius
