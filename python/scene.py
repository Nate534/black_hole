import math
import numpy as np

import python.config as config


class Camera:
	def __init__(self):
		self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
		self.radius = 6.34194e10
		self.minRadius = 1e10
		self.maxRadius = 1e12
		self.azimuth = 0.0
		self.elevation = math.pi / 2.0
		self.orbitSpeed = 0.01
		self.panSpeed = 0.01
		self.zoomSpeed = 25e9
		self.dragging = False
		self.panning = False
		self.moving = False
		self.lastX = 0.0
		self.lastY = 0.0

	@property
	def position(self):
		clamped = max(0.01, min(math.pi - 0.01, self.elevation))
		x = self.radius * math.sin(clamped) * math.cos(self.azimuth)
		y = self.radius * math.cos(clamped)
		z = self.radius * math.sin(clamped) * math.sin(self.azimuth)
		return np.array([x, y, z], dtype=np.float32)

	def update(self):
		self.target[:] = 0.0
		self.moving = bool(self.dragging or self.panning)

	def process_mouse_move(self, x, y):
		dx = float(x - self.lastX)
		dy = float(y - self.lastY)
		if self.dragging and not self.panning:
			self.azimuth += dx * self.orbitSpeed
			self.elevation -= dy * self.orbitSpeed
			self.elevation = max(0.01, min(math.pi - 0.01, self.elevation))
		self.lastX = x
		self.lastY = y
		self.update()

	def process_mouse_button(self, window, button, action, mods):
		if button in (1, 2):
			if action == 1:
				self.dragging = True
				self.panning = False
				self.lastX, self.lastY = window.get_cursor_pos() if hasattr(window, 'get_cursor_pos') else (self.lastX, self.lastY)
			elif action == 0:
				self.dragging = False
				self.panning = False

	def process_scroll(self, xoff, yoff):
		self.radius -= yoff * self.zoomSpeed
		self.radius = max(self.minRadius, min(self.maxRadius, self.radius))
		self.update()


class BlackHole:
	def __init__(self, position, mass):
		self.position = np.array(position, dtype=np.float32)
		self.mass = float(mass)
		self.r_s = 2.0 * config.G * self.mass / (config.c*config.c)


SagA = BlackHole([0.0, 0.0, 0.0], 8.54e36)


def default_objects():
	return [
		[4e11, 0.0, 4e11, 4e10, 1.0,1.0,0.0, 1.98892e30],
		[0.0, 0.0, 0.0, SagA.r_s, 0.0,0.0,0.0, SagA.mass],
	]


def look_at(eye, target, up):
	f = target - eye
	f = f / np.linalg.norm(f)
	s = np.cross(f, up)
	s = s / np.linalg.norm(s)
	u = np.cross(s, f)
	M = np.eye(4, dtype=np.float32)
	M[0, :3] = s
	M[1, :3] = u
	M[2, :3] = -f
	T = np.eye(4, dtype=np.float32)
	T[:3, 3] = -eye
	return M @ T


def perspective(fovy, aspect, znear, zfar):
	f = 1.0 / math.tan(fovy / 2.0)
	M = np.zeros((4,4), dtype=np.float32)
	M[0,0] = f / aspect
	M[1,1] = f
	M[2,2] = (zfar + znear) / (znear - zfar)
	M[2,3] = (2 * zfar * znear) / (znear - zfar)
	M[3,2] = -1.0
	return M

