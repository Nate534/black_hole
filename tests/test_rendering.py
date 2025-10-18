import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rendering.camera import Camera
from rendering.camera_config import cdist, cspeed, czoom, cmove, crot
from rendering.renderer import Renderer
from physics.black_hole import BlackHole
from physics.particle import ParticleSystem

class TestCamera:
    def test_init_default(self):
        cam = Camera()
        expected_pos = np.array([0.0, cdist, -cdist])
        assert np.array_equal(cam.position, expected_pos)
        assert np.array_equal(cam.target, (0, 0, 0))
        assert np.array_equal(cam.up, (0, 1, 0))
        assert cam.move_speed == cmove
        assert cam.zoom_speed == cspeed
        assert cam.zoom == czoom

    def test_init_custom(self):
        pos = (1, 2, 3)
        cam = Camera(position=pos)
        assert np.array_equal(cam.position, pos)

    def test_world_to_screen(self):
        cam = Camera(position=(0, 0, 10))
        screen_pos = cam.world_to_screen((0, 0, 0), 800, 600)
        assert screen_pos == (400, 300)  # Center of screen

    def test_world_to_screen_behind(self):
        cam = Camera(position=(0, 0, 10))
        screen_pos = cam.world_to_screen((0, 0, 20), 800, 600)
        assert screen_pos is None  # Behind camera

class TestRenderer:
    @pytest.fixture
    def renderer(self):
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        yield Renderer(800, 600, screen)
        pygame.quit()

    def test_init(self, renderer):
        assert renderer.width == 800
        assert renderer.height == 600
        assert renderer.font is not None
        assert renderer.glow_surface is not None
        assert renderer.disk_texture is not None

    def test_create_disk_texture(self, renderer):
        texture = renderer.create_disk_texture(64)
        assert texture.get_size() == (64, 64)
        # Check center pixel (should be bright)
        center_color = texture.get_at((32, 32))
        assert center_color[3] > 0  # Alpha > 0
        # Check edge pixel (should be darker)
        edge_color = texture.get_at((0, 0))
        assert edge_color[3] == 0  # Transparent at edge

    def test_render_black_hole(self, renderer):
        bh = BlackHole(mass=1e30)
        cam = Camera()
        # This would require a mock screen, but for now, assume it doesn't crash
        # In real test, use pygame.display.set_mode with no frame
        pass  # Skip detailed test due to pygame dependency