#version 330 core
layout(location = 0) in vec2 aPosition;

uniform vec2 uPosition;
uniform float uRadius;

void main()
{
    vec2 pos = aPosition * uRadius + uPosition;
    gl_Position = vec4(pos, 0.0, 1.0);
}

