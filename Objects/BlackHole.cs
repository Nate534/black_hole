using OpenTK.Mathematics;
using OpenTK.Graphics.OpenGL4;

namespace black_hole.Objects
{
    class BlackHole
    {
        // Black Hole Properties
        public Vector2 Position {get; set;}
        public Vector2 Velocity {get; set;} = Vector2.Zero;
        public double Mass {get; set;}

        public double VisualRadius {get; private set;}

        //constants
        private const double G = 6.67430e-11;
        private const double c = 299792458;

        public BlackHole(Vector2 position, double mass, float scale = 1e9f)
        {
            Position = position;
            Mass = mass;

            double Radius = (2 * G * Mass) / (c * c);

            VisualRadius = (float)(Radius * scale);
        }
    }
}