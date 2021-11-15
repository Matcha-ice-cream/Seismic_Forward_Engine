import taichi as ti

@ti.kernel
def SFE_2mix_show(c:ti.template(), wave:ti.template(), model:ti.template()):
    for i, j in wave:
        c[i, j] = wave[i, j] + model[i, j]/5000.0


@ti.kernel
def SFE_gray_show(c:ti.template(), x:ti.template()):
    for i, j in x:
        c[i, j] = x[i, j] + 3.5/10.0



