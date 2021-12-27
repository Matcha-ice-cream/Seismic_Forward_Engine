import taichi as ti

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












