import taichi as ti

@ti.kernel
def SFE_2mix_show(c:ti.template(), wave:ti.template(), model:ti.template()):
    # 用于混合两个对象，如地震波场与背景模型
    for i, j in wave:
        c[i, j] = wave[i, j] + model[i, j]/5000.0


@ti.kernel
def SFE_gray_show(c:ti.template(), x:ti.template()):
    # 用于将数据转换成灰度图的样式
    for i, j in x:
        c[i, j] = x[i, j] + 3.5/10.0

@ti.kernel
def SFE_wave_render(c: ti.template(), c2: ti.template(), max_l: ti.f32, min_l: ti.f32):
    # 用于将数据绘制\渲染成正演时常见的形式
    for i, j in c:
        d = max_l - min_l
        c2[i, j] = (c[i, j] - min_l) / d

@ti.kernel
def SFE_reverse(c: ti.template(), c_reverse: ti.template(), nx: ti.i32, nz: ti.i32):
    for i, j in c:
        c_reverse[i, j] = c[i, nz - 1 - j]




