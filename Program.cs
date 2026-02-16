using black_hole.rendering;          // rendering access
using black_hole.utils;              // utils access
using OpenTK.Windowing.Desktop;      // 
using OpenTK.Mathematics;            // 

class Program
{
    static void Main()
    {
        var config = Config.Load();

        var nativeSettings = new NativeWindowSettings
        {
            ClientSize = new Vector2i(config.Window.Width, config.Window.Height),
        };

        var blackHoles = new List<BlackHole>
        {
            new BlackHole(new Vector2(-100f, 0f), 5e30), 
            new BlackHole(new Vector2(100f, 0f), 1e31)
        };

        using var window = new SimulationWindow(GameWindowSettings.Default, nativeSettings);
        window.Run();
    }
}
