import glfw
from OpenGL import GL as gl
import math

WIDTH, HEIGHT = 800, 600

c = 299_792_458.0
G = 6.67430e-11

class BlackHole:
	def __init__(self, mass):
		self.mass = mass
		self.r_s = 2*G*mass/(c*c)

bh = BlackHole(8.54e36)

if __name__ == "__main__":
	if not glfw.init():
		raise SystemExit
	win = glfw.create_window(WIDTH, HEIGHT, "2D Lensing (Python)", None, None)
	glfw.make_context_current(win)
	while not glfw.window_should_close(win):
		gl.glClear(gl.GL_COLOR_BUFFER_BIT)
		gl.glMatrixMode(gl.GL_PROJECTION)
		gl.glLoadIdentity()
		gl.glOrtho(-1e11,1e11,-7.5e10,7.5e10,-1,1)
		gl.glMatrixMode(gl.GL_MODELVIEW)
		gl.glLoadIdentity()
		# draw BH circle
		gl.glBegin(gl.GL_TRIANGLE_FAN)
		gl.glColor3f(1.0,0.0,0.0)
		gl.glVertex2f(0.0,0.0)
		for i in range(0,101):
			ang = 2.0*math.pi*i/100
			x = bh.r_s*math.cos(ang)
			y = bh.r_s*math.sin(ang)
			gl.glVertex2f(x,y)
		gl.glEnd()
		glfw.swap_buffers(win)
		glfw.poll_events()
	glfw.terminate()
