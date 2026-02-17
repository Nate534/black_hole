using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using black_hole.Utils;     // Shader compilation

namespace black_hole.Rendering.Shaders
{
    public class BlackHoleShader
    {
        private readonly int _program;
        private readonly int _uPosition;
        private readonly int _uRadius;

        public BlackHoleShader()
        {
            Config? config = ConfigReader.ReadConfig("config.json");
            var shaderPaths = ShaderPaths.Resolve("BlackHole", config.Shaders);
            
            _program = ShaderCompiler.Compile(File.ReadAllText(shaderPaths.vert), File.ReadAllText(shaderPaths.frag));

            config = null;

            _uPosition = GL.GetUniformLocation(_program, "uPosition");
            _uRadius   = GL.GetUniformLocation(_program, "uRadius");
        }

        public void Use() => GL.UseProgram(_program);

        public void SetPosition(Vector2 pos) => GL.Uniform2(_uPosition, pos);

        public void SetRadius(float r) => GL.Uniform1(_uRadius, r);

        public int Program => _program;

    }
}
