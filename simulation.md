# Simulation Documentation

## Overview
This file documents the main simulation script and utility check script.

## main.py

The main entry point for the black hole simulation application. Initializes Pygame, sets up the black hole, particle system, camera, and renderer, then runs the simulation loop.

### Key Components

- **Initialization**:
  - Initializes Pygame and sets up a window (1200x800).
  - Creates a BlackHole with mass 4e37 kg (much larger than default for dramatic effects).
  - Initializes ParticleSystem and Camera with custom position.
  - Creates Renderer for drawing.

- **Particle Setup**:
  - Generates 300 particles for accretion disk between 2.5 and 12 times Schwarzschild radius.
  - Calculates orbital velocities using Keplerian formula \( v = \sqrt{\frac{GM}{r}} \).
  - Adds color variation based on distance (redshift simulation).
  - Adds 30 high-velocity particles for jet effects.

- **Main Loop**:
  - Handles events (quit, escape, gravity toggle, camera reset).
  - Processes continuous camera movement (WASDQE keys).
  - Updates particle system with physics time step (50 seconds per frame for stability).
  - Periodically adds new particles to maintain disk.
  - Renders scene and flips display at 60 FPS.

- **Physics Integration**:
  - Uses Euler method for particle updates.
  - Applies gravitational acceleration from black hole.
  - Changes particle color to black inside horizon.

- **Error Handling**:
  - Catches rendering errors and falls back to simple circle.

This script ties together physics, rendering, and user input for an interactive simulation.

## check.py

A utility script for verifying dependencies and OpenGL capabilities. Currently commented out, but can be used to check if Pygame, NumPy, and OpenGL are properly installed and functioning.

### Functionality

- Imports necessary libraries (Pygame, OpenGL, NumPy).
- Prints versions of Pygame and NumPy.
- Initializes Pygame and creates an OpenGL context.
- Retrieves and prints OpenGL version and GPU information.

Useful for debugging environment setup before running the main simulation.