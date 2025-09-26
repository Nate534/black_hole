"""
Physics module for black hole simulation.

This module contains simulation logic, mathematical models, and integration algorithms.
"""

from .vector3 import Vector3
from .black_hole import BlackHole
from .particle import Particle, ParticleSystem
from .integrator import PhysicsIntegrator

__all__ = ['Vector3', 'BlackHole', 'Particle', 'ParticleSystem', 'PhysicsIntegrator']