# Utils Module Documentation

## Overview
The utils module provides utility functions for vector mathematics used in physics and rendering.

## vectors.py

Contains functions for common vector operations.

- **normalize(vector)**: Normalizes a vector to unit length. If norm is zero, returns the original vector to avoid division by zero.

- **rotate_vector(vector, axis, angle)**: Rotates a vector around an axis by a given angle using Rodrigues' rotation formula: \( v' = v \cos\theta + (k \times v) \sin\theta + k (k \cdot v) (1 - \cos\theta) \), where k is the unit axis.

These functions support 3D transformations in the simulation.

## __init__.py

Marks directory as Python package (empty).