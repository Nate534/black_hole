using OpenTK.Mathematics;
using OpenTK.Graphics.OpenGL4;
using black_hole.Utils;              // Config

namespace black_hole.Objects
{
    public class BlackHole
    {
        public double Mass {get; set;}
        public Vector2 Position {get; set;}
        public Vector2 Velocity {get; set;} = Vector2.Zero;

        public double SchwarzschildRadius { get; private set; }

        public float VisualRadius { get; private set; }

        public BlackHole(double mass, Vector2 position, Vector2 velocity)
        {
            Mass = mass;
            Position = position;
            Velocity = velocity;
            SchwarzschildRadius = GetSchwarzschildRadius();
            VisualRadius = GetVisualRadius(SchwarzschildRadius);
        }

        public float GetSchwarzschildRadius()
        {
            const double G = 6.67430e-11;
            const double C = 299792458;
            return (float)((2 * G * Mass) / (C * C));
        }

        private float GetVisualRadius(double radius)
        {
            Config? config = ConfigReader.ReadConfig("config.json");
            float VisualRadius = (float)(Math.Log10(radius / config.Simulation.ReferenceRadius) * config.Simulation.VisualScale);
            config = null;

            return VisualRadius;
        }
    }
}