"""
GPU compositor built on ModernGL.

Each frame the CPU uploads three textures (camera image, UI layer, glow
layer) and a single full-screen shader combines them:

1. Draw the camera image, with its original colours, inside the camera
   rectangle. Everything outside it is black.
2. Draw glass panels (the chart and the message box) as rounded rectangles
   with a soft tint and a hairline border.
3. Add the glow layer, sampled from several mipmap levels, to get a bloom
   effect without extra render passes.
4. Blend the UI layer on top.
"""

import moderngl
import numpy as np
import pygame

MAX_PANELS = 8

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_cam;
uniform sampler2D u_ui;
uniform sampler2D u_glow;

uniform vec2  u_res;
uniform vec4  u_cam_rect;        // x, y, width, height of the camera area
uniform vec2  u_cam_scale;       // crop that keeps the camera aspect ratio
uniform vec2  u_cam_offset;
uniform float u_cam_ready;       // 0 while no frame has arrived yet

uniform int   u_panel_count;
uniform vec4  u_panels[8];       // x, y, width, height in pixels
uniform vec4  u_panel_accent[8]; // rgb colour, a = strength
uniform float u_radius;

in vec2 v_uv;
out vec4 f_color;

const vec3 BACKGROUND = vec3(0.0);

// Signed distance to a rounded rectangle centred on the origin.
float sd_round_rect(vec2 p, vec2 half_size, float r) {
    vec2 q = abs(p) - half_size + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

void main() {
    // Textures are uploaded top row first, so flip v to get top-left origin.
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec2 px = uv * u_res;

    // Camera, untouched, inside its rectangle; black everywhere else.
    vec2 local = (px - u_cam_rect.xy) / u_cam_rect.zw;
    bool in_cam = all(greaterThanEqual(local, vec2(0.0))) &&
                  all(lessThan(local, vec2(1.0)));
    vec3 col = BACKGROUND;
    if (in_cam && u_cam_ready > 0.5) {
        col = texture(u_cam, u_cam_offset + local * u_cam_scale).rgb;
    }

    for (int i = 0; i < u_panel_count; ++i) {
        vec4 r = u_panels[i];
        vec2 centre = r.xy + r.zw * 0.5;
        float sd = sd_round_rect(px - centre, r.zw * 0.5, u_radius);

        // Panel fill: a dark smoky tint, a little lighter at the top.
        float inside = 1.0 - smoothstep(-0.8, 0.8, sd);
        float t = clamp((px.y - r.y) / r.w, 0.0, 1.0);   // 0 at top, 1 at bottom
        vec3 glass = mix(col, vec3(0.075, 0.082, 0.105), 0.78);
        glass += vec3(0.030) * (1.0 - t);
        col = mix(col, glass, inside);

        // Hairline border: brighter at the top, like light hitting glass.
        float edge = 1.0 - smoothstep(0.0, 1.1, abs(sd + 0.55));
        float top = mix(0.20, 0.07, t);
        vec4 acc = u_panel_accent[i];
        vec3 border = mix(vec3(1.0), acc.rgb, acc.a);
        col = mix(col, border, edge * (top + acc.a * 0.6));

        // Accent also lights the inside edge a little.
        float inner = (1.0 - smoothstep(0.0, 14.0, -sd)) * inside;
        col += acc.rgb * inner * acc.a * 0.10;
    }

    // Bloom: sum of three blur levels of the glow layer.
    vec3 glow = textureLod(u_glow, uv, 1.0).rgb * 0.55
              + textureLod(u_glow, uv, 2.6).rgb * 0.75
              + textureLod(u_glow, uv, 4.2).rgb * 0.85;
    col += glow * (1.0 - col * 0.5);

    vec4 ui = texture(u_ui, uv);
    col = mix(col, ui.rgb, ui.a);

    f_color = vec4(col, 1.0);
}
"""


class Compositor:
    def __init__(self, ctx: moderngl.Context, size: tuple, camera_rect: tuple):
        self.ctx = ctx
        self.size = size
        self.camera_rect = camera_rect
        self.program = ctx.program(vertex_shader=VERTEX_SHADER,
                                   fragment_shader=FRAGMENT_SHADER)

        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._vbo = ctx.buffer(quad.tobytes())
        self._vao = ctx.vertex_array(self.program, [(self._vbo, "2f", "in_pos")])

        self._cam_tex = None
        self._cam_size = (0, 0)
        self._ui_tex = self._make_texture(size, 4, mipmaps=False)
        self._glow_tex = self._make_texture(size, 3, mipmaps=True)

        self.program["u_cam"] = 0
        self.program["u_ui"] = 1
        self.program["u_glow"] = 2
        self.program["u_res"] = size
        self.program["u_radius"] = 14.0
        self.program["u_cam_ready"] = 0.0
        self.program["u_cam_scale"] = (1.0, 1.0)
        self.program["u_cam_offset"] = (0.0, 0.0)
        self.program["u_cam_rect"] = tuple(float(v) for v in camera_rect)

        # A 1x1 black texture so the shader is valid before the camera starts.
        self._set_camera_texture(np.zeros((1, 1, 3), dtype=np.uint8))

    def _make_texture(self, size, components, mipmaps):
        tex = self.ctx.texture(size, components)
        tex.alignment = 1
        if mipmaps:
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        else:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = False
        return tex

    def _set_camera_texture(self, frame_rgb: np.ndarray) -> None:
        h, w = frame_rgb.shape[:2]
        if (w, h) != self._cam_size:
            if self._cam_tex is not None:
                self._cam_tex.release()
            self._cam_tex = self._make_texture((w, h), 3, mipmaps=True)
            self._cam_size = (w, h)
        self._cam_tex.write(np.ascontiguousarray(frame_rgb))
        self._cam_tex.build_mipmaps()

    # -- per-frame inputs ------------------------------------------------------

    def update_camera(self, frame_rgb: np.ndarray) -> None:
        self._set_camera_texture(frame_rgb)
        self.program["u_cam_ready"] = 1.0
        scale, offset = cover_crop(self._cam_size, self.camera_rect[2:])
        self.program["u_cam_scale"] = scale
        self.program["u_cam_offset"] = offset

    def update_layers(self, ui: pygame.Surface, glow: pygame.Surface) -> None:
        self._ui_tex.write(pygame.image.tobytes(ui, "RGBA"))
        self._glow_tex.write(pygame.image.tobytes(glow, "RGB"))
        self._glow_tex.build_mipmaps()

    def set_panels(self, panels) -> None:
        panels = panels[:MAX_PANELS]
        rects = np.zeros((MAX_PANELS, 4), dtype="f4")
        accents = np.zeros((MAX_PANELS, 4), dtype="f4")
        for i, p in enumerate(panels):
            rects[i] = (p.rect.x, p.rect.y, p.rect.w, p.rect.h)
            c = p.accent_color
            accents[i] = (c[0] / 255, c[1] / 255, c[2] / 255, p.accent)
        self.program["u_panels"].write(rects.tobytes())
        self.program["u_panel_accent"].write(accents.tobytes())
        self.program["u_panel_count"] = len(panels)

    def render(self) -> None:
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, *self.size)
        self._cam_tex.use(0)
        self._ui_tex.use(1)
        self._glow_tex.use(2)
        self._vao.render(moderngl.TRIANGLE_STRIP)


def cover_crop(src_size, dst_size):
    """
    UV scale and offset that fill `dst_size` with `src_size` without
    stretching, cropping the longer side equally on both ends (like CSS
    `object-fit: cover`).
    """
    sw, sh = src_size
    dw, dh = dst_size
    src_aspect = sw / sh
    dst_aspect = dw / dh
    if src_aspect > dst_aspect:
        sx = dst_aspect / src_aspect
        return (sx, 1.0), ((1.0 - sx) / 2.0, 0.0)
    sy = src_aspect / dst_aspect
    return (1.0, sy), (0.0, (1.0 - sy) / 2.0)
