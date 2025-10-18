# Physics Module Documentation

## Overview
The physics module implements the core simulation logic for a black hole and particle system using Newtonian gravity. It includes classes for black holes, particles, and particle systems, along with physical constants.

## black_hole.py

The `BlackHole` class simulates a black hole in 3D space, calculating gravitational effects and event horizon.

### Class: BlackHole

- **__init__(self, mass, position = (0,0,0))**: Initializes the black hole with mass (kg) and position (m). Computes Schwarzschild radius \( r_s = \frac{2GM}{c^2} \).

- **get_rad_vec(self, position)**: Returns vector from black hole to given position.

- **get_rad(self, r_vec)**: Calculates radial distance from black hole.

- **calc_grav_accel(self, position)**: Computes gravitational acceleration at position using \( a = \frac{GM}{r^2} \), directed toward black hole. Handles r=0 case.

- **is_inside_horizon(self, position)**: Checks if position is within Schwarzschild radius.

This class provides gravitational field calculations for the simulation.

## constants.py

Defines key physical constants:

- **G = 6.67430e-11**: Gravitational constant (m³ kg⁻¹ s⁻²).

- **C = 299792458.0**: Speed of light (m/s).

- **bhmass = 5.972e30**: Default black hole mass (kg, ~3 solar masses).

- **bhinitpos = (0, 0, 0)**: Default black hole position.

Used in gravitational and relativistic calculations.

## particle.py

Implements particle dynamics under gravity.

### Class: Particle

- **__init__(self, mass, position, velocity, colour)**: Initializes particle with mass, position, velocity, color. Maintains trail of positions for rendering.

- **update(self, acceleration, dt)**: Updates velocity and position via Euler integration. Manages trail length.

### Class: ParticleSystem

- **__init__(self)**: Empty particle list, gravity enabled.

- **add_particle(self, mass, position, velocity, colour)**: Adds particle to system.

- **update(self, black_hole, dt)**: Updates all particles with gravitational acceleration from black hole. Changes color if inside horizon.

Manages collection of particles and their interactions.

## __init__.py

Marks directory as Python package (empty).