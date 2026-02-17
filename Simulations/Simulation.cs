using black_hole.Objects;            // BlackHole
using black_hole.Utils;              // Config

namespace black_hole.Simulations
{
    public class Simulation
    {
        public List<BlackHole> BlackHoles { get; set; }

        public Simulation(List<BlackHole> blackHoles)
        {
            BlackHoles = blackHoles;
        }
    }
}