import moderngl
import numpy as np
from PIL import Image

def shade_it(size, shader, shader_args=None):
  ctx = moderngl.create_standalone_context()

  prog = ctx.program(
      vertex_shader='''
          #version 330

          in vec2 in_vert;
          in vec2 in_uv;
          out vec2 f_uv;
          void main() {
              f_uv = in_uv;
              gl_Position = vec4(in_vert, 0.0, 1.0);
          }
      ''',
      fragment_shader=shader,
  )

  samplers = []
  if shader_args:
    for key, value in shader_args.items():
      if isinstance(value, Image.Image):
        tex = ctx.texture(value.size, len(value.mode), value.tobytes())
        sam = ctx.sampler(texture=tex)
        value = len(samplers)
        samplers.append(sam.assign(value))

      ref = prog.get(key, None)
      if ref is not None:
        ref.value = value
      else:
        print(f'unused argument: {key}')

  square = np.array([
       1.0,  1.0,   1.0, 0.0,
      -1.0,  1.0,   0.0, 0.0,
       1.0, -1.0,   1.0, 1.0,
      -1.0, -1.0,   0.0, 1.0,
  ])

  square = ctx.buffer(square.astype('f4').tobytes())
  square = ctx.simple_vertex_array(prog, square, 'in_vert', 'in_uv')

  output = ctx.simple_framebuffer(size)
  output.use()
  output.clear(0.0, 1.0, 0.0, 1.0)

  square.scope = ctx.scope(framebuffer=output, samplers=samplers)
  square.render(moderngl.TRIANGLE_STRIP)

  return Image.frombytes('RGBA', output.size, output.read(components=4), 'raw', 'RGBA', 0, -1)
