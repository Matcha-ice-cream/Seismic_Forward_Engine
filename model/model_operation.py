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

        self.model_rand = ti.Vector.field(2, dtype=ti.f32, shape=(20, 20))

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
            # assert "please check your file!/请检查文件路径！(来自喵子emm的善意提醒)"
            pass

    @ti.func
    def fade(self, t):
        return 6.0 * t ** 5.0 - 15.0 * t ** 4.0 + 10.0 * t ** 3.0

    @ti.kernel
    def model_perlin(self, lx: ti.i32, lz: ti.i32):
        if self.nx / lx != 0 or self.nz / lz != 0:
            # assert "Please make sure that the small grid is divisible./请确保小格子可被整除.(来自喵子emm的善意提醒)"
            pass
        # ti.root.dense(ti.ij, (self.nx / lx + 1, self.nz / lz + 1)).place(self.model_rand)
        for i, j in self.model_rand:
            a = (ti.random(ti.f32)-0.5)*2
            b = (ti.random(ti.f32)-0.5)*2
            self.model_rand[i, j] = ti.Vector([a, b]).normalized()

        for i, j in self.model_vp:
            xn = int(ti.floor(i / lx))
            zn = int(ti.floor(j / lz))
            xi = i % lx
            zi = j % lz
            xf = float(xi) / float(lx)
            zf = float(zi) / float(lz)
            xt = self.fade(xf)
            zt = self.fade(zf)
            Pa = ti.Vector([xf, zf])
            Pb = ti.Vector([xf - 1.0, zf])
            Pc = ti.Vector([xf, zf - 1.0])
            Pd = ti.Vector([xf - 1.0, zf - 1.0])
            TA = self.model_rand[xn, zn].dot(Pa)
            TB = self.model_rand[xn + 1, zn].dot(Pb)
            TC = self.model_rand[xn, zn + 1].dot(Pc)
            TD = self.model_rand[xn + 1, zn + 1].dot(Pd)

            l1 = TA + (TB - TA) * xt
            l2 = TC + (TD - TC) * xt
            u = l1 + (l2 - l1) * zt

            self.model_vp[i, j] = u
            self.model_vs[i, j] = u
            self.model_rho[i, j] = u


    @ti.kernel
    def model_perlin_change(self):
        pass

