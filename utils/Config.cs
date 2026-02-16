using System.Text.Json;
using System.IO;

namespace black_hole.utils
{
    public class WindowConfig
    {
        public int Width { get; set; }
        public int Height { get; set; }
    }

    public class Config
    {
        public WindowConfig Window { get; set; } = new WindowConfig();

        public static Config Load(string path = "utils/configs/config.json")
        {
            if (!File.Exists(path))
                return new Config();

            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<Config>(json) ?? new Config();
        }
    }
}