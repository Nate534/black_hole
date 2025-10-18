import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics.black_hole import BlackHole
from physics.particle import Particle, ParticleSystem
from physics.constants import G, C

class TestBlackHole:
    def test_init(self):
        bh = BlackHole(mass=1e30, position=(1, 2, 3))
        assert bh.mass == 1e30
        assert np.array_equal(bh.position, (1, 2, 3))
        expected_rs = 2 * G * 1e30 / (C ** 2)
        assert bh.schwarz_radius == expected_rs

    def test_get_rad_vec(self):
        bh = BlackHole(mass=1e30, position=(0, 0, 0))
        r_vec = bh.get_rad_vec((3, 4, 0))
        assert np.array_equal(r_vec, (-3, -4, 0))

    def test_get_rad(self):
        bh = BlackHole(mass=1e30, position=(0, 0, 0))
        r = bh.get_rad((3, 4, 0))
        assert r == 5.0

    def test_calc_grav_accel(self):
        bh = BlackHole(mass=1e30, position=(0, 0, 0))
        accel = bh.calc_grav_accel((5, 0, 0))
        expected_mag = G * 1e30 / 25
        assert abs(np.linalg.norm(accel) - expected_mag) < 1e-10
        assert np.array_equal(accel, (-expected_mag, 0, 0))  # Direction toward BH

    def test_calc_grav_accel_at_center(self):
        bh = BlackHole(mass=1e30, position=(0, 0, 0))
        accel = bh.calc_grav_accel((0, 0, 0))
        assert np.array_equal(accel, (0, 0, 0))

    def test_is_inside_horizon(self):
        bh = BlackHole(mass=1e30)
        assert bh.is_inside_horizon((0, 0, 0))  # At center
        assert not bh.is_inside_horizon((2000, 0, 0))  # Outside

class TestParticle:
    def test_init(self):
        p = Particle(mass=1e10, position=(1, 2, 3), velocity=(0, 0, 0), colour=(1, 0, 0))
        assert p.mass == 1e10
        assert np.array_equal(p.position, (1, 2, 3))
        assert np.array_equal(p.velocity, (0, 0, 0))
        assert p.colour == (1, 0, 0)
        assert len(p.trail) == 0

    def test_update_vector_accel(self):
        p = Particle(mass=1e10, position=(0, 0, 0), velocity=(1, 0, 0), colour=(1, 0, 0))
        accel = np.array([0, 1, 0])
        p.update(accel, 1.0)
        assert np.array_equal(p.velocity, (1, 1, 0))
        assert np.array_equal(p.position, (1, 1, 0))
        assert len(p.trail) == 1

    def test_update_scalar_accel(self):
        p = Particle(mass=1e10, position=(3, 0, 0), velocity=(0, 0, 0), colour=(1, 0, 0))
        p.update(1.0, 1.0)  # Scalar accel toward center
        expected_vel = np.array([-1, 0, 0])  # Direction normalized
        assert np.allclose(p.velocity, expected_vel, atol=1e-10)

    def test_trail_management(self):
        p = Particle(mass=1e10, position=(0, 0, 0), velocity=(0, 0, 0), colour=(1, 0, 0))
        for i in range(60):
            p.update((0, 0, 0), 1.0)
        assert len(p.trail) == 50  # Max length

class TestParticleSystem:
    def test_init(self):
        ps = ParticleSystem()
        assert len(ps.particles) == 0
        assert ps.grav_enabled

    def test_add_particle(self):
        ps = ParticleSystem()
        ps.add_particle(1e10, (0, 0, 0), (0, 0, 0), (1, 0, 0))
        assert len(ps.particles) == 1
        assert ps.particles[0].mass == 1e10

    def test_update_with_gravity(self):
        bh = BlackHole(mass=1e30, position=(0, 0, 0))
        ps = ParticleSystem()
        p = Particle(mass=1e10, position=(5, 0, 0), velocity=(0, 0, 0), colour=(1, 0, 0))
        ps.add_particle(p.mass, p.position, p.velocity, p.colour)
        ps.update(bh, 1.0)
        # Particle should accelerate toward BH
        assert ps.particles[0].velocity[0] < 0

    def test_update_without_gravity(self):
        bh = BlackHole(mass=1e30, position=(0, 0, 0))
        ps = ParticleSystem()
        ps.grav_enabled = False
        p = Particle(mass=1e10, position=(0, 0, 0), velocity=(1, 0, 0), colour=(1, 0, 0))
        ps.add_particle(p.mass, p.position, p.velocity, p.colour)
        ps.update(bh, 1.0)
        # No acceleration
        assert np.array_equal(ps.particles[0].velocity, (1, 0, 0))

    def test_horizon_color_change(self):
        bh = BlackHole(mass=1e30, position=(0, 0, 0))  # Large BH
        ps = ParticleSystem()
        p = Particle(mass=1e10, position=(0, 0, 0), velocity=(0, 0, 0), colour=(1, 0, 0))
        ps.add_particle(p.mass, p.position, p.velocity, p.colour)
        ps.update(bh, 1.0)
        assert ps.particles[0].colour == (0, 0, 0)  # Black inside horizon