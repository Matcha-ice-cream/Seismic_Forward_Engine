import taichi as ti

ti.init(arch=ti.gpu)


@ti.kernel
def ra():
    for i in range(100):
        a = ti.random(ti.f32)
        print(a)



ra()
