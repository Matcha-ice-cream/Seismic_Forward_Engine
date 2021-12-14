import taichi as ti
import numpy as np


@ti.data_oriented
class getmodel:
    def __init__(self, nx, nz, dx, dz):
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.model_vp = ti.field(dtype=ti.f32, shape=(nx, nz))
        self.model_vs = ti.field(dtype=ti.f32, shape=(nx, nz))
        self.model_rho = ti.field(dtype=ti.f32, shape=(nx, nz))

        self.model_vx1 = ti.field(dtype=ti.f32, shape=(nx, nz))
        self.model_vx2 = ti.field(dtype=ti.f32, shape=(nx, nz))

        self.model_vz1 = ti.field(dtype=ti.f32, shape=(nx, nz))
        self.model_vz2 = ti.field(dtype=ti.f32, shape=(nx, nz))

    def diff_1(self, v):
        for i, j in v:
            self.model_vx1[i, j] = (v[i + 1, j] - v[i - 1, j]) / (2 * self.dx)
            self.model_vz1[i, j] = (v[i, j + 1] - v[i, j - 1]) / (2 * self.dz)

    def diff_2(self, v):
        for i, j in v:
            self.model_vx2[i, j] = (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / (self.dx ** 2)
            self.model_vz2[i, j] = (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / (self.dz ** 2)

    @ti.kernel
    def model_default(self):
        for i, j in self.model_vp:
            if j < self.nz / 3:
                self.model_vp[i, j] = 2500.0
                self.model_vs[i, j] = 2500.0
            else:
                self.model_vp[i, j] = 2000.0
                self.model_vs[i, j] = 2000.0
        for i, j in self.model_rho:
            if j < self.nz / 3:
                self.model_rho[i, j] = 1.0
            else:
                self.model_rho[i, j] = 1.0
        for i, j in self.model_vp:
            self.model_vx1[i, j] = (self.model_vp[i + 1, j] - self.model_vp[i - 1, j]) / (2 * self.dx)
            self.model_vz1[i, j] = (self.model_vp[i, j + 1] - self.model_vp[i, j - 1]) / (2 * self.dz)
        for i, j in self.model_vp:
            self.model_vx2[i, j] = (self.model_vp[i + 1, j] - 2 * self.model_vp[i, j] + self.model_vp[i - 1, j]) / (
                        self.dx ** 2)
            self.model_vz2[i, j] = (self.model_vp[i, j + 1] - 2 * self.model_vp[i, j] + self.model_vp[i, j - 1]) / (
                        self.dz ** 2)

    def model_file(self, path, mod):
        if mod == 'vp':
            arr = np.loadtxt(path)
            self.model_vp = arr.from_numpy()
            self.diff_1(self.model_vp)
            self.diff_2(self.model_vp)
        if mod == 'rho':
            arr = np.loadtxt(path)
            self.model_rho = arr.from_numpy()
        if mod == 'vs':
            arr = np.loadtxt(path)
            self.model_vs = arr.from_numpy()
        else:
            assert "please check your file!"

    @ti.func
    def fade(self, t):
        return 6 * t ** 5 + 15 * t ** 4 + 10 * t ** 3

    @ti.kernel
    def model_perlin(self):
        

        pass
