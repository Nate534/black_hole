import numpy as np
from .constants import G, C

class Particle:
    def __init__(self, mass, position, velocity, colour):
        self.mass = mass
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        
        self.colour = colour
        self.trail = []
        self.max_trail_length = 50

    def update(self, acceleration, dt):
        acceleration = np.array(acceleration, dtype=np.float64)
        if acceleration.shape == ():
            if np.linalg.norm(self.position) > 0:
                direction = -self.position / np.linalg.norm(self.position)
                acceleration = acceleration * direction
            else:
                acceleration = np.array([0.0, 0.0, 0.0])

        elif acceleration.shape != self.velocity.shape:
            if len(acceleration) < 3:
                acceleration = np.pad(acceleration, (0, 3 - len(acceleration)))
            else:
                acceleration = acceleration[:3]
        
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        self.trail.append(np.copy(self.position))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)

class ParticleSystem:

    def __init__(self):
        self.particles = []
        self.grav_enabled = True

    def add_particle(self, mass, position, velocity, colour):
        self.particles.append(Particle(mass, position, velocity, colour))

    def update(self, black_hole, dt):
        for particle in self.particles:
            if self.grav_enabled:
                accel = black_hole.calc_grav_accel(particle.position)
                particle.update(accel, dt)
            else:
                particle.update(np.array([0.0, 0.0, 0.0]), dt)
            
            pos = np.array(particle.position, dtype=np.float64)

            if black_hole.is_inside_horizon(pos):
                particle.colour = (0, 0, 0)
