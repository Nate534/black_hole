import os

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADERS = os.path.join(ROOT, "python", "shaders")

# Window and render sizes
WIDTH, HEIGHT = 800, 600
COMPUTE_STATIC_W, COMPUTE_STATIC_H = 200, 150
COMPUTE_DYNAMIC_W, COMPUTE_DYNAMIC_H = 100, 75

# Physical constants
c = 299_792_458.0
G = 6.67430e-11

# Tuning
FOV_DEGREES = 60.0
NEAR_PLANE = 1e9
FAR_PLANE = 1e14
MAX_OBJECTS = 16
