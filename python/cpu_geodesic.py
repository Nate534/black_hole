import math
import numpy as np

c = 299_792_458.0
G = 6.67430e-11

class BlackHole:
	def __init__(self, mass):
		self.mass = mass
		self.r_s = 2.0 * G * mass / (c*c)

class Camera:
	def __init__(self):
		self.pos = np.array([0.0, 0.0, 6.34194e10], dtype=np.float64)
		self.target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
		self.fovY = 60.0

class Ray:
	def __init__(self, pos, dir, rs):
		x,y,z = pos
		dx,dy,dz = dir
		self.x, self.y, self.z = x,y,z
		self.r = math.sqrt(x*x+y*y+z*z)
		self.theta = math.acos(z/self.r)
		self.phi = math.atan2(y, x)
		self.dr     = math.sin(self.theta)*math.cos(self.phi)*dx + math.sin(self.theta)*math.sin(self.phi)*dy + math.cos(self.theta)*dz
		self.dtheta = (math.cos(self.theta)*math.cos(self.phi)*dx + math.cos(self.theta)*math.sin(self.phi)*dy - math.sin(self.theta)*dz) / self.r
		self.dphi   = (-math.sin(self.phi)*dx + math.cos(self.phi)*dy) / (self.r * math.sin(self.theta))
		self.L = self.r*self.r*math.sin(self.theta)*self.dphi
		f = 1.0 - rs / self.r
		dt_dlam = math.sqrt((self.dr*self.dr)/f + self.r*self.r*(self.dtheta*self.dtheta + (math.sin(self.theta)**2)*(self.dphi*self.dphi)))
		self.E = f * dt_dlam
		self.rs = rs

	def step(self, dlam):
		if self.r <= self.rs: return
		def rhs():
			r,th,ph, dr,dth,dph = self.r,self.theta,self.phi, self.dr,self.dtheta,self.dphi
			f = 1.0 - self.rs/r
			dt_dlam = self.E / f
			dr2 = -(self.rs/(2*r*r))*f*(dt_dlam**2) + (self.rs/(2*r*r*f))*(dr*dr) + r*(dth*dth + (math.sin(th)**2)*(dph*dph))
			dth2 = -(2.0/r)*dr*dth + math.sin(th)*math.cos(th)*(dph*dph)
			dph2 = -(2.0/r)*dr*dph - 2.0*math.cos(th)/math.sin(th)*dth*dph
			return np.array([dr, dth, dph, dr2, dth2, dph2], dtype=np.float64)
		k1 = rhs()
		state = np.array([self.r,self.theta,self.phi,self.dr,self.dtheta,self.dphi], dtype=np.float64)
		k2 = rhs()
		k3 = rhs()
		k4 = rhs()
		state += (dlam/6.0)*(k1 + 2*k2 + 2*k3 + k4)
		self.r,self.theta,self.phi,self.dr,self.dtheta,self.dphi = state
		self.x = self.r*math.sin(self.theta)*math.cos(self.phi)
		self.y = self.r*math.sin(self.theta)*math.sin(self.phi)
		self.z = self.r*math.cos(self.theta)

if __name__ == "__main__":
	bh = BlackHole(8.54e36)
	cam = Camera()
	forward = (cam.target - cam.pos) / np.linalg.norm(cam.target - cam.pos)
	r = Ray(cam.pos, forward, bh.r_s)
	for i in range(10000):
		r.step(1e7)
		if r.r > 1e14 or r.r <= bh.r_s:
			break
	print(f"ended at r={r.r:.3e}")
