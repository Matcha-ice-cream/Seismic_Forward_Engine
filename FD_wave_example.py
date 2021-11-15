import taichi as ti
from FD_wave.wave_module_2d4d import wave
from FD_wave.receiver_module import receiver
import Visualization.SFE_visual as vis

ti.init(arch=ti.gpu)
frame = 1

c = ti.field(dtype=ti.f32, shape=(600,600))
c_s = ti.field(dtype=ti.f32, shape=(1000, 1000))
wave_cs = wave(300, 400, 600, 600, 10.0, 10.0, 1, 50, 1, 1e-3, 3, 20)
wave_cs.mod_default()
wave_cs.PML_cal()
gui = ti.GUI("wave", (600, 600))


receiver_cs = receiver('PIC', 1000, 1000)
receiver_cs.rec_init(600, 600)
gui_rec = ti.GUI("rec", (1000, 1000))



while frame < 1000:
    wave_cs.wave_field_cal(frame)
    
    receiver_cs.rec_gather(wave_cs.p, int(frame/1))
    receiver_cs.rec_dynamic(wave_cs.dt, int(frame/1), 12.0)

    vis.SFE_2mix_show(c, wave_cs.p, wave_cs.model_v)
    vis.SFE_gray_show(c_s, receiver_cs.rec_value)
    gui.set_image(c)
    gui_rec.set_image(c_s)


    gui.show()
    gui_rec.show()

    frame += 1

path='./data/rec_seis.txt'
receiver_cs.export(receiver_cs.rec_value, path)
