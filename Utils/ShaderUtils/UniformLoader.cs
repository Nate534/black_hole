using OpenTK.Graphics.OpenGL4;

namespace black_hole.Utils
{
    public static class UniformLoader
    {
        public static int Load(int shader, string name)
        {
            int loc = GL.GetUniformLocation(shader, name);
            if (loc == -1)
                throw new Exception($"Uniform '{name}' not found.");
            return loc;
        }
    }  
}