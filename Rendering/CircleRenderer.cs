using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using System;

namespace black_hole.Rendering
{
    public class CircleRenderer
    {
        private int _vao;
        private int _vbo;
        private int _shader;

        public CircleRenderer(int shader)
        {
            _shader = shader;
            CreateCircle();
        }

        private void CreateCircle()
        {
            int segments = 64;
            float[] vertices = new float[(segments + 2) * 2];

            vertices[0] = 0f;
            vertices[1] = 0f;

            for (int i = 0; i <= segments; i++)
            {
                double angle = i * 2.0 * Math.PI / segments;
                vertices[(i + 1) * 2] = (float)Math.Cos(angle);
                vertices[(i + 1) * 2 + 1] = (float)Math.Sin(angle);
            }

            _vao = GL.GenVertexArray();
            _vbo = GL.GenBuffer();

            GL.BindVertexArray(_vao);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _vbo);
            GL.BufferData(BufferTarget.ArrayBuffer, vertices.Length * sizeof(float), vertices, BufferUsageHint.StaticDraw);
            GL.EnableVertexAttribArray(0);
            GL.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, 2 * sizeof(float), 0);
        }

        public void Draw(Vector2 position, float radius)
        {
            GL.UseProgram(_shader);

            GL.Uniform2(GL.GetUniformLocation(_shader, "uPosition"), position);
            GL.Uniform1(GL.GetUniformLocation(_shader, "uRadius"), radius);

            GL.BindVertexArray(_vao);
            GL.DrawArrays(PrimitiveType.TriangleFan, 0, 66);
        }
    }
}