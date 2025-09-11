import sys
import os
import math
import ctypes
import numpy as np

import glfw
from OpenGL import GL as gl

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADERS = os.path.join(ROOT, "python", "shaders")

WIDTH, HEIGHT = 800, 600
COMPUTE_W, COMPUTE_H = 200, 150

# Globals
Gravity = False
c = 299_792_458.0
G = 6.67430e-11

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
		global Gravity
		if button in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_MIDDLE):
			if action == glfw.PRESS:
				self.dragging = True
				self.panning = False
				self.lastX, self.lastY = glfw.get_cursor_pos(window)
			elif action == glfw.RELEASE:
				self.dragging = False
				self.panning = False
		if button == glfw.MOUSE_BUTTON_RIGHT:
			if action == glfw.PRESS:
				Gravity = True
			elif action == glfw.RELEASE:
				Gravity = False
	
	def process_scroll(self, xoff, yoff):
		self.radius -= yoff * self.zoomSpeed
		self.radius = max(self.minRadius, min(self.maxRadius, self.radius))
		self.update()

camera = Camera()

class BlackHole:
	def __init__(self, position, mass):
		self.position = np.array(position, dtype=np.float32)
		self.mass = float(mass)
		self.r_s = 2.0 * G * self.mass / (c*c)

SagA = BlackHole([0.0, 0.0, 0.0], 8.54e36)

class ObjectData(ctypes.Structure):
	_fields_ = [
		("posRadius", ctypes.c_float * 4),
		("color", ctypes.c_float * 4),
		("mass", ctypes.c_float),
		("_pad", ctypes.c_float * 3),
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

class Engine:
	def __init__(self):
		if not glfw.init():
			raise RuntimeError("GLFW init failed")
		glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
		glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
		glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
		self.window = glfw.create_window(WIDTH, HEIGHT, "Black Hole (Python)", None, None)
		if not self.window:
			glfw.terminate()
			raise RuntimeError("Failed to create window")
		glfw.make_context_current(self.window)
		glfw.swap_interval(1)
		# callbacks
		glfw.set_window_user_pointer(self.window, None)
		glfw.set_mouse_button_callback(self.window, self._mouse_button_cb)
		glfw.set_cursor_pos_callback(self.window, self._cursor_pos_cb)
		glfw.set_scroll_callback(self.window, self._scroll_cb)
		glfw.set_key_callback(self.window, self._key_cb)
		# programs
		self.quad_vao, self.texture = self._create_quad()
		self.present_prog = self._create_present_program()
		self.grid_prog = self._create_program_from_files(
			os.path.join(SHADERS, "grid.vert"),
			os.path.join(SHADERS, "grid.frag"),
		)
		self.compute_prog = self._create_compute_program(os.path.join(SHADERS, "geodesic.comp"))
		# UBOs
		self.camera_ubo = gl.glGenBuffers(1)
		gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.camera_ubo)
		gl.glBufferData(gl.GL_UNIFORM_BUFFER, 128, None, gl.GL_DYNAMIC_DRAW)
		gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, 1, self.camera_ubo)
		self.disk_ubo = gl.glGenBuffers(1)
		gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.disk_ubo)
		gl.glBufferData(gl.GL_UNIFORM_BUFFER, 16, None, gl.GL_DYNAMIC_DRAW)
		gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, 2, self.disk_ubo)
		self.objects_ubo = gl.glGenBuffers(1)
		gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.objects_ubo)
		obj_ubo_size = ctypes.sizeof(ctypes.c_int) + 3*ctypes.sizeof(ctypes.c_float) + 16*(4*4+4*4+4)
		gl.glBufferData(gl.GL_UNIFORM_BUFFER, obj_ubo_size, None, gl.GL_DYNAMIC_DRAW)
		gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, 3, self.objects_ubo)
		# grid mesh
		self.grid_vao = 0
		self.grid_vbo = 0
		self.grid_ebo = 0
		self.grid_index_count = 0
	
	def _mouse_button_cb(self, window, button, action, mods):
		camera.process_mouse_button(window, button, action, mods)
	
	def _cursor_pos_cb(self, window, x, y):
		camera.process_mouse_move(x, y)
	
	def _scroll_cb(self, window, xoff, yoff):
		camera.process_scroll(xoff, yoff)
	
	def _key_cb(self, window, key, scancode, action, mods):
		global Gravity
		if action == glfw.PRESS and key == glfw.KEY_G:
			Gravity = not Gravity
			print(f"[INFO] Gravity {'ON' if Gravity else 'OFF'}")
	
	def _create_quad(self):
		quad = np.array([
			-1.0,  1.0,  0.0, 1.0,
			-1.0, -1.0,  0.0, 0.0,
			 1.0, -1.0,  1.0, 0.0,
			-1.0,  1.0,  0.0, 1.0,
			 1.0, -1.0,  1.0, 0.0,
			 1.0,  1.0,  1.0, 1.0,
		], dtype=np.float32)
		vao = gl.glGenVertexArrays(1)
		vbo = gl.glGenBuffers(1)
		gl.glBindVertexArray(vao)
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
		gl.glBufferData(gl.GL_ARRAY_BUFFER, quad.nbytes, quad, gl.GL_STATIC_DRAW)
		gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 16, ctypes.c_void_p(0))
		gl.glEnableVertexAttribArray(0)
		gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, False, 16, ctypes.c_void_p(8))
		gl.glEnableVertexAttribArray(1)
		tex = gl.glGenTextures(1)
		gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
		gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, COMPUTE_W, COMPUTE_H, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
		return vao, tex
	
	def _compile(self, src, stage):
		shader = gl.glCreateShader(stage)
		gl.glShaderSource(shader, src)
		gl.glCompileShader(shader)
		ok = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
		if not ok:
			log = gl.glGetShaderInfoLog(shader).decode()
			raise RuntimeError(f"Shader compile error: {log}")
		return shader
	
	def _create_present_program(self):
		vs = """
		#version 330 core
		layout (location = 0) in vec2 aPos;
		layout (location = 1) in vec2 aTexCoord;
		out vec2 TexCoord;
		void main(){ gl_Position = vec4(aPos, 0.0, 1.0); TexCoord = aTexCoord; }
		"""
		fs = """
		#version 330 core
		in vec2 TexCoord; out vec4 FragColor; uniform sampler2D screenTexture;
		void main(){ FragColor = texture(screenTexture, TexCoord); }
		"""
		v = self._compile(vs, gl.GL_VERTEX_SHADER)
		f = self._compile(fs, gl.GL_FRAGMENT_SHADER)
		prog = gl.glCreateProgram()
		gl.glAttachShader(prog, v)
		gl.glAttachShader(prog, f)
		gl.glLinkProgram(prog)
		gl.glDeleteShader(v)
		gl.glDeleteShader(f)
		return prog
	
	def _read_shader_text(self, path):
		# Read shader text, strip BOM and leading whitespace before #version
		with open(path, "rb") as f:
			data = f.read()
		text = data.decode("utf-8-sig")
		text = text.lstrip()
		idx = text.find("#version")
		if idx > 0:
			text = text[idx:]
		return text

	def _create_program_from_files(self, vert_path, frag_path):
		vs = self._read_shader_text(vert_path)
		fs = self._read_shader_text(frag_path)
		v = self._compile(vs, gl.GL_VERTEX_SHADER)
		f = self._compile(fs, gl.GL_FRAGMENT_SHADER)
		prog = gl.glCreateProgram()
		gl.glAttachShader(prog, v)
		gl.glAttachShader(prog, f)
		gl.glLinkProgram(prog)
		if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
			raise RuntimeError(gl.glGetProgramInfoLog(prog).decode())
		gl.glDeleteShader(v)
		gl.glDeleteShader(f)
		return prog
	
	def _create_compute_program(self, path):
		src = self._read_shader_text(path)
		cs = self._compile(src, gl.GL_COMPUTE_SHADER)
		prog = gl.glCreateProgram()
		gl.glAttachShader(prog, cs)
		gl.glLinkProgram(prog)
		if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
			raise RuntimeError(gl.glGetProgramInfoLog(prog).decode())
		gl.glDeleteShader(cs)
		return prog
	
	def generate_grid(self, objects):
		gridSize = 25
		spacing = 1e10
		verts = []
		indices = []
		for z in range(gridSize+1):
			for x in range(gridSize+1):
				worldX = (x - gridSize/2) * spacing
				worldZ = (z - gridSize/2) * spacing
				y = 0.0
				for obj in objects:
					objPos = np.array(obj[0:3], dtype=np.float64)
					mass = float(obj[7])
					radius = float(obj[3])
					rs = 2.0 * G * mass / (c*c)
					dx = worldX - objPos[0]
					dz = worldZ - objPos[2]
					dist = math.sqrt(dx*dx + dz*dz)
					if dist > rs:
						deltaY = 2.0 * math.sqrt(rs * (dist - rs))
						y += float(deltaY) - 3e10
					else:
						y += 2.0 * math.sqrt(rs*rs) - 3e10
				verts.extend([worldX, y, worldZ])
		for z in range(gridSize):
			for x in range(gridSize):
				i = z * (gridSize + 1) + x
				indices.extend([i, i+1, i, i+gridSize+1])
		verts = np.array(verts, dtype=np.float32)
		indices = np.array(indices, dtype=np.uint32)
		if self.grid_vao == 0:
			self.grid_vao = gl.glGenVertexArrays(1)
			self.grid_vbo = gl.glGenBuffers(1)
			self.grid_ebo = gl.glGenBuffers(1)
		gl.glBindVertexArray(self.grid_vao)
		gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.grid_vbo)
		gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_DYNAMIC_DRAW)
		gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.grid_ebo)
		gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
		gl.glEnableVertexAttribArray(0)
		gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 12, ctypes.c_void_p(0))
		self.grid_index_count = indices.size
		gl.glBindVertexArray(0)
	
	def draw_grid(self, viewProj):
		gl.glUseProgram(self.grid_prog)
		loc = gl.glGetUniformLocation(self.grid_prog, "viewProj")
		gl.glUniformMatrix4fv(loc, 1, False, viewProj.astype(np.float32))
		gl.glBindVertexArray(self.grid_vao)
		gl.glDisable(gl.GL_DEPTH_TEST)
		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
		gl.glDrawElements(gl.GL_LINES, self.grid_index_count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))
		gl.glBindVertexArray(0)
		gl.glEnable(gl.GL_DEPTH_TEST)
	
	def upload_camera(self):
		pos = camera.position
		fwd = (camera.target - pos)
		fwd = fwd / np.linalg.norm(fwd)
		up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
		right = np.cross(fwd, up)
		right = right / np.linalg.norm(right)
		up = np.cross(right, fwd)
		data = np.zeros(32, dtype=np.float32)
		data[0:3] = pos
		data[4:7] = right
		data[8:11] = up
		data[12:15] = fwd
		data[16] = math.tan(math.radians(60.0*0.5))
		data[17] = WIDTH/HEIGHT
		data[18] = 1.0 if (camera.dragging or camera.panning) else 0.0
		gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.camera_ubo)
		gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, data.nbytes, data)
	
	def upload_disk(self):
		r1 = SagA.r_s * 2.2
		r2 = SagA.r_s * 5.2
		num = 2.0
		thk = 1e9
		data = np.array([r1, r2, num, thk], dtype=np.float32)
		gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.disk_ubo)
		gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, data.nbytes, data)
	
	def upload_objects(self, objects):
		# objects: list of (x,y,z,r, cr,cg,cb, mass)
		count = min(len(objects), 16)
		# pack into std140-like layout
		posRadius = np.zeros((16,4), dtype=np.float32)
		color = np.zeros((16,4), dtype=np.float32)
		mass = np.zeros(16, dtype=np.float32)
		for i in range(count):
			x,y,z,r, cr,cg,cb, m = objects[i]
			posRadius[i] = [x,y,z,r]
			color[i] = [cr,cg,cb,1.0]
			mass[i] = m
		header = np.array([count,0,0,0], dtype=np.float32)
		blob = header.tobytes() + posRadius.tobytes() + color.tobytes() + mass.tobytes()
		gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.objects_ubo)
		gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, len(blob), blob)
	
	def dispatch_compute(self):
		cw = 100 if camera.moving else 200
		ch = 75 if camera.moving else 150
		# re-alloc texture only if size changed
		gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
		current_w = gl.glGetTexLevelParameteriv(gl.GL_TEXTURE_2D, 0, gl.GL_TEXTURE_WIDTH)
		current_h = gl.glGetTexLevelParameteriv(gl.GL_TEXTURE_2D, 0, gl.GL_TEXTURE_HEIGHT)
		if current_w != cw or current_h != ch:
			gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, cw, ch, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
		gl.glUseProgram(self.compute_prog)
		self.upload_camera()
		self.upload_disk()
		self.upload_objects(objects)
		gl.glBindImageTexture(0, self.texture, 0, False, 0, gl.GL_WRITE_ONLY, gl.GL_RGBA8)
		groupsX = int(math.ceil(cw / 16.0))
		groupsY = int(math.ceil(ch / 16.0))
		gl.glDispatchCompute(groupsX, groupsY, 1)
		gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
	
	def draw_fullscreen(self):
		gl.glUseProgram(self.present_prog)
		gl.glBindVertexArray(self.quad_vao)
		gl.glActiveTexture(gl.GL_TEXTURE0)
		gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
		loc = gl.glGetUniformLocation(self.present_prog, "screenTexture")
		gl.glUniform1i(loc, 0)
		gl.glDisable(gl.GL_DEPTH_TEST)
		gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
		gl.glEnable(gl.GL_DEPTH_TEST)

objects = [
	[4e11, 0.0, 4e11, 4e10, 1.0,1.0,0.0, 1.98892e30],
	[0.0, 0.0, 0.0, SagA.r_s, 0.0,0.0,0.0, SagA.mass],
]


def main():
	engine = Engine()
	last_time = glfw.get_time()
	while not glfw.window_should_close(engine.window):
		gl.glViewport(0, 0, WIDTH, HEIGHT)
		gl.glClearColor(0.0, 0.0, 0.0, 1.0)
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
		# grid
		engine.generate_grid(objects)
		eye = camera.position
		view = look_at(eye, camera.target, np.array([0,1,0], dtype=np.float32))
		proj = perspective(math.radians(60.0), COMPUTE_W/COMPUTE_H, 1e9, 1e14)
		engine.draw_grid((proj @ view).T)
		# compute + present
		engine.dispatch_compute()
		engine.draw_fullscreen()
		glfw.swap_buffers(engine.window)
		glfw.poll_events()
	glfw.destroy_window(engine.window)
	glfw.terminate()

if __name__ == "__main__":
	main()
