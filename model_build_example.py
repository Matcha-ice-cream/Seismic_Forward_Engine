import taichi as ti
from model.model_operation import getmodel
import Tools.SFE_tools as tools

ti.init(arch=ti.gpu)

model_cs = getmodel(500, 500, 10, 10)

model_cs.model_perlin(50, 50)

gui = ti.GUI("model", (500, 500))

# tools.export(model_cs.model_vp, "data.txt")
while True:
    gui.set_image(model_cs.model_vp)
    gui.show()