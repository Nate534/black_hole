# Rendering Module Documentation

## Overview
The rendering module handles visualization of the black hole simulation using Pygame. It includes camera controls, rendering of black holes, particles, and UI elements.

## camera.py

The `Camera` class manages the viewpoint and user controls for navigating the 3D simulation in 2D screen space.

### Class: Camera

- **__init__(self, position=None)**: Initializes camera at default or given position, with target at origin and up vector (0,1,0). Sets movement, zoom, rotation speeds from config.

- **handle_event(self, event, dt)**: Processes discrete events like key presses for reset (R), zoom in/out (+/-).

- **handle_continuous_movement(self, keys, dt)**: Handles continuous key presses (WASDQE) for movement. Calculates world-space movement vectors, applies velocity with damping, updates position. Prevents camera from getting too close to black hole.

- **world_to_screen(self, world_pos, screen_width, screen_height)**: Projects 3D world position to 2D screen coordinates using simple perspective projection based on camera forward vector and zoom.

- **normalize(vector)**: Utility function to normalize vectors, handling zero norm case.

Provides smooth camera movement and projection for rendering.

## camera_config.py

Defines camera configuration parameters:

- **cdist = 1e15**: Default distance from black hole.

- **cspeed = 1e15**: Movement speed.

- **czoom = 0.5**: Zoom speed multiplier.

- **cmove = 5e13**: Move speed.

- **crot = 1**: Rotation speed.

These are used to tune camera behavior.

## renderer.py

The `Renderer` class handles all drawing operations, including black hole, particles, trails, glow effects, and UI.

### Class: Renderer

- **__init__(self, width, height, screen)**: Initializes with screen dimensions, font, glow surface, and precomputes accretion disk texture.

- **create_disk_texture(self, size)**: Generates a radial gradient texture for the accretion disk, with colors varying by distance (red to blue).

- **render(self, black_hole, particle_system, camera)**: Main render loop: clears screen, renders black hole, particles, applies glow, renders UI.

- **render_black_hole(self, black_hole, camera)**: Renders black hole with accretion disk (scaled texture), photon sphere, event horizon (dark circle), shadow, and lensing rings.

- **render_particles(self, particle_system, camera)**: Renders each particle as a circle with size based on mass, glow effect, and trail if present.

- **render_trail(self, particle, camera)**: Draws particle trail as fading lines on glow surface.

- **apply_glow(self)**: Blurs glow surface and blends onto main screen for lighting effects.

- **render_ui(self, camera, particle_system, black_hole)**: Displays camera position, zoom, particle count, gravity status, black hole mass, Schwarzschild radius, and controls help.

Manages visual representation with effects like glow and trails.

## __init__.py

Marks directory as Python package (empty).