using OpenTK.Graphics.OpenGL4;
using black_hole.Rendering.Shaders;            // BlackHole

namespace black_hole.Rendering
{
    public class BHRenderer
    {
        private int _vao;
        private int _vbo;
        private int _vertexCount = 32;

        private readonly BlackHoleShader _shader;
   

        public BHRenderer(BlackHoleShader shader)
        {
            _shader = shader;
            CreateCircle();
        }

        private void CreateCircle()
        {
            float[] vertices = new float[(_vertexCount + 2) * 2];

            vertices[0] = 0f;
            vertices[1] = 0f;

            for (int i = 0; i <= _vertexCount; i++)
            {
                double angle = i * 2.0 * Math.PI / _vertexCount;
                vertices[(i + 1) * 2] = (float)Math.Cos(angle);
                vertices[(i + 1) * 2 + 1] = (float)Math.Sin(angle);
            }

            _vao = GL.GenVertexArray();
            _vbo = GL.GenBuffer();

            GL.BindVertexArray(_vao);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _vbo);
            GL.BufferData(
                BufferTarget.ArrayBuffer, 
                vertices.Length * sizeof(float), 
                vertices, 
                BufferUsageHint.StaticDraw
            );

            GL.EnableVertexAttribArray(0);
            GL.VertexAttribPointer(
                0, 
                2, 
                VertexAttribPointerType.Float, 
                false, 
                2 * sizeof(float), 
                0
            );
        }

        public void RenderBH(List<Objects.BlackHole> blackHoles)
        {
            _shader.Use();
            GL.BindVertexArray(_vao);

            foreach (var blackHole in blackHoles)
            {
                _shader.SetPosition(blackHole.Position);
                _shader.SetRadius(blackHole.VisualRadius);

                GL.DrawArrays(PrimitiveType.TriangleFan, 0, _vertexCount + 2);
            }
        }
    } 
}