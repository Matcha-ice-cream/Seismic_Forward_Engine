import taichi as ti
from model.model_operation import getmodel
import Tools.SFE_tools as tools
import Visualization.SFE_visual as visual

ti.init(arch=ti.gpu)

c_reverse = ti.field(ti.f32, shape = (500, 500))

model_cs = getmodel(500, 500, 10, 10)

model_cs.model_perlin_munk(50, 50, 1000.0, 1000.0, 1500.0)

gui = ti.GUI("model", (500, 500))
c2 = ti.field(ti.f32, shape=(500, 500))

visual.SFE_wave_render(model_cs.model_vp, c2, 1500, 1530)

# tools.export(model_cs.model_vp, "data.txt")
# print(c2)
while True:
    visual.SFE_reverse(c2, c_reverse, 500, 500)
    gui.set_image(c_reverse)
    gui.show()