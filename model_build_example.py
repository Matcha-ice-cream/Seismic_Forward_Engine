import taichi as ti
from model.model_operation import getmodel

ti.init(arch=ti.gpu)

model_cs = getmodel(500, 500, 10, 10)


model_cs.model_perlin(50, 50)

gui = ti.GUI("model", (500, 500))


while True:
    gui.set_image(model_cs.model_vp)

    gui.show()