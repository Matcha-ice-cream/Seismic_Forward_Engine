from re import S
import taichi as ti
from model.model_operation import getmodel
import Tools.SFE_tools as tools
import Visualization.SFE_visual as visual
import Tools.SFE_math as Smath

ti.init(arch=ti.gpu)

c_reverse = ti.field(ti.f32, shape = (500, 500))
pi = 3.1415926535
model_cs = getmodel(500, 500, 10, 10)
model_cs.generate_rand()

model_cs1 = getmodel(500,500,10,10)
model_cs1.generate_rand()


gui = ti.GUI("model", (500, 500))
c2 = ti.field(ti.f32, shape=(500, 500))
c3 = ti.field(ti.f32, shape=(500, 500))
model_cs.model_perlin_change(50,50,1000.0,1000.0,1500.0,pi/100)
model_cs1.model_perlin_change(25,25,1000.0,1000.0,1500.0,pi/50)
Smath.combine(model_cs.model_vp,model_cs1.model_vp,c2)
tools.export(c2, "cs.txt")
# tools.export(model_cs.model_vp, "demodata_perlin.txt")
# print(model_cs.model_vp)
while True:
    model_cs.model_perlin_change(50,50,1000.0,1000.0,1500.0,pi/75)
    model_cs1.model_perlin_change(10,10,1000.0,1000.0,1500.0,pi/50)
    Smath.combine(model_cs.model_vp,model_cs1.model_vp,c2)
    visual.SFE_wave_render(c2, c3, 1490, 1540)
    visual.SFE_reverse(c3, c_reverse, 500, 500)
    gui.set_image(c_reverse)
    # c2 = model_cs.model_vp + model_cs1.model_vp
    # gui.set_image(model_cs1.model_vp)
    gui.show()

