using black_hole.utils;           // Config
using OpenTK.Graphics.OpenGL4;    // GL
using OpenTK.Mathematics;         // Vector2i, Color4
using OpenTK.Windowing.Desktop;   // GameWindow, GameWindowSettings, NativeWindowSettings
using OpenTK.Windowing.Common;
using black_hole.Objects;    // FrameEventArgs, ResizeEventArgs

namespace black_hole.Rendering
{
    public class SimulationWindow : GameWindow
    {

        private readonly List<BlackHole> _blackHoles;

        public SimulationWindow(GameWindowSettings gameSettings, NativeWindowSettings nativeSettings)
            :base(gameSettings, nativeSettings)
        {
            _blackHoles = blackHoles;
        }

        protected override void OnLoad()
        {
            base.OnLoad();
            GL.ClearColor(Color4.Red);
        }

        protected override void OnRenderFrame(FrameEventArgs args)
        {
            base.OnRenderFrame(args);

            GL.Clear(ClearBufferMask.ColorBufferBit);
            GL.PushMatrix();
            GL.Translate(Size.x / 2f, Size.y / 2f, 0f);

            foreach (var bh in _blackHoles)
            {
                bh.Draw();
            }

            GL.PopMatrix();
            SwapBuffers();
        }

        protected override void OnResize(ResizeEventArgs e)
        {
            base.OnResize(e);
            GL.Viewport(0, 0, e.Width, e.Height);
        }
    }
}
