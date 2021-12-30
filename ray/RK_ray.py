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
    def RK(self, X, v, v_dx, v_dz):
        Xp = ti.Vector([0.0, 0.0, 0.0, 0.0])
        x0 = X[0]
        z0 = X[1]
        px0 = X[2]
        pz0 = X[3]

        ll1 = px0
        ll2 = pz0





















