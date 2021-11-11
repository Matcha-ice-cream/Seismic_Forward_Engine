import taichi as ti
from FD_wave.wave_module_2d4d import wave

ti.init(arch=ti.gpu)

wave_cs = wave(300, 400, 600, 600, 10.0, 10.0, 1, 30, 1, 1e-3, 3, 20)

frame = 1.0
wave_cs.mod_default()
wave_cs.PML_cal()

gui = ti.GUI("wave", (600, 600))

while frame < 10000:
    wave_cs.wave_field_cal(frame)

    gui.set_image(wave_cs.p)
    gui.show()
    frame += 1.0
