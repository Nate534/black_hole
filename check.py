# # check.py
# import pygame
# from pygame.locals import *  # This imports DOUBLEBUF, OPENGL, and other constants
# from OpenGL.GL import *
# from OpenGL.GLU import *
# import numpy as np

# print("All imports successful!")
# print(f"Pygame version: {pygame.version.ver}")
# print(f"NumPy version: {np.__version__}")

# # Try to get OpenGL information
# pygame.init()
# display = (800, 600)
# pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

# print(f"OpenGL version: {glGetString(GL_VERSION).decode()}")
# print(f"GPU: {glGetString(GL_RENDERER).decode()}")

# pygame.quit()