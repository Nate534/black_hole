import math
import numpy as np

import glfw
from OpenGL import GL as gl

import python.config as config
from python.scene import look_at, perspective
from python.renderer import Engine


def main():
	engine = Engine()
	try:
		while not glfw.window_should_close(engine.window):
			gl.glViewport(0, 0, config.WIDTH, config.HEIGHT)
			gl.glClearColor(0.0, 0.0, 0.0, 1.0)
			gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
			# grid
			engine.generate_grid()
			eye = engine.camera.position
			view = look_at(eye, engine.camera.target, np.array([0,1,0], dtype=np.float32))
			proj = perspective(math.radians(config.FOV_DEGREES), config.COMPUTE_STATIC_W/config.COMPUTE_STATIC_H, config.NEAR_PLANE, config.FAR_PLANE)
			engine.draw_grid((proj @ view).T)
			# compute + present
			engine.dispatch_compute()
			engine.draw_fullscreen()
			glfw.swap_buffers(engine.window)
			glfw.poll_events()
	finally:
		engine.destroy()


if __name__ == "__main__":
	main()
