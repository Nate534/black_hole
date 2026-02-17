using black_hole.Objects;            // BlackHole
using black_hole.Utils;              // Config

namespace black_hole.Simulations
{
    public static class SimFactory
    {
        public static Simulation CreateFromConfig(Config config)
        {
            var blackHoles = config.BlackHoles
                .Select(data => new BlackHole(data.Mass, data.Position.ToVector2(), data.Velocity.ToVector2()))
                .ToList();

            return new Simulation(blackHoles);
        }
    }
}