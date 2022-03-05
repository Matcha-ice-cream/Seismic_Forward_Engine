import taichi as ti


@ti.kernel
def combine(a:ti.template(), b:ti.template(), c:ti.template()):
    for i, j in a:
        c[i,j]=a[i,j]/2.0+b[i,j]/2.0

