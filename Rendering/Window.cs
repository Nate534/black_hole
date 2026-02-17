using OpenTK.Graphics.OpenGL4;    // GL
using OpenTK.Mathematics;         // Vector2i, Color4
using OpenTK.Windowing.Desktop;   // GameWindow, GameWindowSettings, NativeWindowSettings
using OpenTK.Windowing.Common;
using black_hole.Simulations;     // Simulation
using black_hole.Utils;           // Config
using black_hole.Rendering.Shaders; // BlackHoleShader

namespace black_hole.Rendering
{
    public class Window : GameWindow
    {
        private readonly Simulation simulation;
        private BHRenderer bhRenderer;
        // private readonly RayRenderer rayRenderer;

        public Window(GameWindowSettings gameSettings, NativeWindowSettings nativeSettings, Simulation simulation)
        :base(gameSettings, nativeSettings)
        {
            this.simulation = simulation;
            this.bhRenderer = new BHRenderer(new BlackHoleShader());
        }

        protected override void OnLoad()
        {
            base.OnLoad();
            GL.ClearColor(Color4.Red);
        }

        protected override void OnRenderFrame(FrameEventArgs args)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            bhRenderer.RenderBH(simulation.BlackHoles);
            // rayRenderer.RenderRays(simulation.Rays);
            
            SwapBuffers();
        }

        protected override void OnResize(ResizeEventArgs e)
        {
            base.OnResize(e);
            GL.Viewport(0, 0, e.Width, e.Height);
        }
    }
}
