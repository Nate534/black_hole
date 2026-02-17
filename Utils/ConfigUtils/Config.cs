using System.Text.Json;
using OpenTK.Mathematics;     // Vector2

namespace black_hole.Utils
{
    public class Config
    {
        public WindowConfig Window { get; set; } = null!;
        public SimulationConfig Simulation { get; set; } = null!;
        public List<BlackHoleData> BlackHoles { get; set; } = new List<BlackHoleData>();
        public ShaderConfig Shaders { get; set; } = new ShaderConfig();
    }

    public class WindowConfig
    {
        public int Width { get; set; }
        public int Height { get; set; }
    }

    public class SimulationConfig
    {
        public float VisualScale { get; set; }
        public float ReferenceRadius { get; set; }
    }

    public class Vector2Data
    {
        public float X { get; set; }
        public float Y { get; set; }

        public Vector2 ToVector2() => new Vector2(X, Y);
    }


    public class BlackHoleData
    {
        public double Mass { get; set; }
        public Vector2Data Position { get; set; } = default!;
        public Vector2Data Velocity { get; set; } = new Vector2Data();
    }

    public class ShaderConfig : Dictionary<string, ShaderPair>
    {
    }

    public class ShaderPair
    {
        public string Vert { get; set; } = null!;
        public string Frag { get; set; } = null!;
    }

    public static class ShaderPaths
    {
        private static string BasePath = "black_hole/Rendering/Shaders/";

        public static (string vert, string frag) Resolve(string name, ShaderConfig config)
        {
            if (!config.TryGetValue(name, out var shader))
                throw new KeyNotFoundException($"Shader not found in config: {name}");

            return (
                Path.Combine(BasePath, shader.Vert),
                Path.Combine(BasePath, shader.Frag)
            );
        }
    }

    public static class ConfigReader
    {
        public static Config ReadConfig(string path = "Utils/ConfigUtils/config.json")
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Config file not found: {path}");

            string json = File.ReadAllText(path);

            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            var config = JsonSerializer.Deserialize<Config>(json, options);

            if (config == null)
                throw new System.Exception("Failed to parse config.json");

            return config;
        }
    }
}