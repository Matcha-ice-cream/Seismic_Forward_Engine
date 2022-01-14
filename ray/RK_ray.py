import taichi as ti
import Tools.SFE_math as Smath

@ti.data_oriented
class RK_ray:
    def __init__(self, src_x, src_z, nx, nz, dx, dz, ray_n):
        self.src_x = src_x
        self.src_z = src_z
        self.nx = nx
        self.nz = nz
        self.dz = dz
        self.dx = dx
        self.ray_n = ray_n

    @ti.func
    def dis(self, x1, y1, x2, y2):
        return ((x1 - x2)**2.0 + (y1 - y2)**2.0) ** 0.5


    @ti.func
    def RK(self, X, v):
        Xp = ti.Vector([0.0, 0.0, 0.0, 0.0])
        x0 = X[0]
        z0 = X[1]
        px0 = X[2]
        pz0 = X[3]

        i = ti.round(x0 / self.dx)
        j = ti.round(z0 / self.dz)












        ll1 = px0
        ll2 = pz0






















