import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics.black_hole import BlackHole
from physics.particle import ParticleSystem
from rendering.camera import Camera
from rendering.renderer import Renderer

class TestIntegration:
    def test_black_hole_particle_interaction(self):
        bh = BlackHole(mass=1e30, position=(0, 0, 0))
        ps = ParticleSystem()
        ps.add_particle(1e10, (5, 0, 0), (0, 0, 0), (1, 0, 0))
        
        initial_pos = ps.particles[0].position.copy()
        ps.update(bh, 1.0)
        
        # Particle should move toward black hole
        assert ps.particles[0].position[0] < initial_pos[0]

    def test_camera_projection(self):
        cam = Camera(position=(0, 0, 10))
        bh = BlackHole(mass=1e30, position=(0, 0, 0))
        
        screen_pos = cam.world_to_screen(bh.position, 800, 600)
        assert screen_pos == (400, 300)

    def test_full_simulation_step(self):
        bh = BlackHole(mass=1e30)
        ps = ParticleSystem()
        ps.add_particle(1e10, (5, 0, 0), (0, 0, 0), (1, 0, 0))
        cam = Camera()
        
        # Simulate one update step
        ps.update(bh, 1.0)
        
        # Check particle moved
        assert not np.array_equal(ps.particles[0].position, (5, 0, 0))
        
        # Camera should still work
        screen_pos = cam.world_to_screen(ps.particles[0].position, 800, 600)
        assert screen_pos is not None