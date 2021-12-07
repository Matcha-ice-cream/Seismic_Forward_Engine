import taichi as ti
import numpy as np

@ti.data_oriented
class getmodel:
    def __init__(self, nx, nz):
        self.nx = nx
        self.nz = nz
        self.model_vp = ti.field(dtype=ti.f32, shape=(nx, nz))
        self.model_vs = ti.field(dtype=ti.f32, shape=(nx, nz))
        self.model_rho = ti.field(dtype=ti.f32, shape=(nx, nz))

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

    def model_file(self, path, mod):
        if mod == 'vp':
            arr = np.loadtxt(path)
            self.model_vp = arr.from_numpy()
        if mod == 'rho':
            arr = np.loadtxt(path)
            self.model_rho = arr.from_numpy()
        if mod == 'vs':
            arr = np.loadtxt(path)
            self.model_vs = arr.from_numpy()
        else:
            assert "please check your file!"
