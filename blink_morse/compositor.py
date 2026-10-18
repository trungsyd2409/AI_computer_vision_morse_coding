"""
GPU compositor built on ModernGL.

Each frame the CPU uploads two textures (camera image and UI layer) and a
full-screen shader puts the UI on top of the camera. The camera keeps its
own colours: no grading, dimming, vignette or blur.
"""

import moderngl
import numpy as np
import pygame

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
uniform vec2  u_cam_scale;       // crop that keeps the camera aspect ratio
uniform vec2  u_cam_offset;
uniform float u_cam_ready;       // 0 while no frame has arrived yet

in vec2 v_uv;
out vec4 f_color;

void main() {
    // Textures are uploaded top row first, so flip v to get top-left origin.
    vec2 uv  = vec2(v_uv.x, 1.0 - v_uv.y);
    vec3 col = texture(u_cam, u_cam_offset + uv * u_cam_scale).rgb;
    col = mix(vec3(0.06), col, u_cam_ready);
    vec4 ui = texture(u_ui, uv);
    f_color = vec4(mix(col, ui.rgb, ui.a), 1.0);
}
"""


class Compositor:
    def __init__(self, ctx: moderngl.Context, size: tuple):
        self.ctx = ctx
        self.size = size
        self.program = ctx.program(vertex_shader=VERTEX_SHADER,
                                   fragment_shader=FRAGMENT_SHADER)

        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._vbo = ctx.buffer(quad.tobytes())
        self._vao = ctx.vertex_array(self.program, [(self._vbo, "2f", "in_pos")])

        self._cam_tex = None
        self._cam_size = (0, 0)
        self._ui_tex = self._make_texture(size, 4)

        self.program["u_cam"] = 0
        self.program["u_ui"] = 1
        self.program["u_cam_ready"] = 0.0
        self.program["u_cam_scale"] = (1.0, 1.0)
        self.program["u_cam_offset"] = (0.0, 0.0)

        # A 1x1 black texture so the shader is valid before the camera starts.
        self._set_camera_texture(np.zeros((1, 1, 3), dtype=np.uint8))

    def _make_texture(self, size, components):
        tex = self.ctx.texture(size, components)
        tex.alignment = 1
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = False
        return tex

    def _set_camera_texture(self, frame_rgb: np.ndarray) -> None:
        h, w = frame_rgb.shape[:2]
        if (w, h) != self._cam_size:
            if self._cam_tex is not None:
                self._cam_tex.release()
            self._cam_tex = self._make_texture((w, h), 3)
            self._cam_size = (w, h)
        self._cam_tex.write(np.ascontiguousarray(frame_rgb))

    # -- per-frame inputs ------------------------------------------------------

    def update_camera(self, frame_rgb: np.ndarray) -> None:
        self._set_camera_texture(frame_rgb)
        self.program["u_cam_ready"] = 1.0
        scale, offset = cover_crop(self._cam_size, self.size)
        self.program["u_cam_scale"] = scale
        self.program["u_cam_offset"] = offset

    def update_ui(self, ui: pygame.Surface) -> None:
        self._ui_tex.write(pygame.image.tobytes(ui, "RGBA"))

    def render(self) -> None:
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, *self.size)
        self._cam_tex.use(0)
        self._ui_tex.use(1)
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
