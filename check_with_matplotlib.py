import sys
sys.path.insert(0, 'src')
import matplotlib.pyplot as plt
import numpy as np
from physics.black_hole import BlackHole
from physics.particle import ParticleSystem
from physics.constants import G

# Create black hole
black_hole = BlackHole(mass=4e37, position=(0, 0, 0))

# Create particle system
particle_system = ParticleSystem()

# Add some particles for testing
num_particles = 10
inner_radius = black_hole.schwarz_radius * 2.5
outer_radius = black_hole.schwarz_radius * 12

for i in range(num_particles):
    distance = np.random.uniform(inner_radius, outer_radius)
    angle = np.random.uniform(0, 2 * np.pi)
    x = distance * np.cos(angle)
    z = distance * np.sin(angle)
    y = np.random.uniform(-outer_radius/50, outer_radius/50)
    
    orbital_speed = np.sqrt(G * black_hole.mass / distance)
    vx = -orbital_speed * np.sin(angle)
    vz = orbital_speed * np.cos(angle)
    
    particle_system.add_particle(
        mass=np.random.uniform(1e9, 1e10),
        position=(x, y, z),
        velocity=(vx, 0, vz),
        colour=(1, 0, 0)
    )

# Update a few times
for _ in range(10):
    particle_system.update(black_hole, 50.0)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot black hole
ax.scatter(0, 0, 0, color='black', s=100, label='Black Hole')

# Plot particles
for particle in particle_system.particles:
    ax.scatter(particle.position[0], particle.position[1], particle.position[2],
               color=particle.colour, s=10)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Black Hole Simulation Positions')
plt.savefig('simulation.png')
print("Plot saved to simulation.png")