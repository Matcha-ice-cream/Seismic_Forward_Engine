import taichi as ti
from wave_module import wave

ti.init(arch=ti.gpu)


wave_cs = wave(200, 300, 400, 400, 10.0, 10.0, 1, 10, 1, 1e-3, 3, 20)

frame = 1.0
wave_cs.mod_default()
wave_cs.PML_pro(20, 20, 4.0, 2000.0, 1.0)

gui = ti.GUI("wave", (400, 400))

while frame < 50000:
    wave_cs.wave_field_cal(frame)
    gui.set_image(wave_cs.p_field)
    gui.show()
    frame += 1.0
