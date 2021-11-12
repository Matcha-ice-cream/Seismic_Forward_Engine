import taichi as ti
from FD_wave.wave_module_2d4d import wave
from FD_wave.receiver_module import receiver

ti.init(arch=ti.gpu)
frame = 1


wave_cs = wave(300, 400, 600, 600, 10.0, 10.0, 1, 30, 1, 1e-3, 3, 20)
wave_cs.mod_default()
wave_cs.PML_cal()
gui = ti.GUI("wave", (600, 600))


receiver_cs = receiver('int', 100, 1000)
receiver_cs.rec_default(600, 600)
gui_rec = ti.GUI("rec", (100, 1000))



while frame < 1000:
    wave_cs.wave_field_cal(frame)
    receiver_cs.rec_gather(600, 600, wave_cs.p, frame)

    gui.set_image(wave_cs.p)
    gui.show()

    gui_rec.set_image(receiver_cs.rec_value)
    gui_rec.show()

    frame += 1
