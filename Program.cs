using black_hole.Utils;  
using black_hole.Simulations;    // Simulation factory
using black_hole.Rendering; 
using OpenTK.Windowing.Desktop;   // GameWindowSettings, NativeWindowSettings
using OpenTK.Mathematics;

class Program
{
    static void Main()
    {
        var config = ConfigReader.ReadConfig();

        var simulation = SimFactory.CreateFromConfig(config);

        var gameSettings = GameWindowSettings.Default;
        var nativeSettings = new NativeWindowSettings
        {
            ClientSize = new Vector2i(config.Window.Width, config.Window.Height),
            Title = "Black Hole Simulation"
        };

        config = null;

        using var window = new Window(gameSettings, nativeSettings, simulation);
        window.Run();
    }
}
