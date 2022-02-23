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

        self.model_munk = ti.field(dtype=ti.f32, shape=(nx, nz))

        self.rand_array = ti.Vector.field(2, dtype=ti.f32, shape=(100, 100))

        self.data = ti.field(dtype=ti.f32, shape=3)


        self.output_d = ti.Vector([0.0, 0.0, 0.0])
        self.output = ti.Vector([0.0, 0.0, 0.0])

    def diff_1(self, v):
        for i, j in v:
            self.model_vx1[i, j] = (v[i + 1, j] - v[i - 1, j]) / (2 * self.dx)
            self.model_vz1[i, j] = (v[i, j + 1] - v[i, j - 1]) / (2 * self.dz)

    def diff_2(self, v):
        for i, j in v:
            self.model_vx2[i, j] = (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / (self.dx ** 2)
            self.model_vz2[i, j] = (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / (self.dz ** 2)

    @ti.kernel
    def generate_rand(self):
        for i, j in self.rand_array:
            a = (ti.random(ti.f32) - 0.5) * 2
            b = (ti.random(ti.f32) - 0.5) * 2
            self.rand_array[i, j] = ti.Vector([a, b]).normalized()

    
    @ti.func
    def fade(self, t):
        return 6.0 * t ** 5.0 - 15.0 * t ** 4.0 + 10.0 * t ** 3.0
    
    @ti.func
    def node(self, i, j, lx, lz, z0, v0, B):
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
        TA = self.rand_array[xn, zn].dot(Pa)
        TB = self.rand_array[xn + 1, zn].dot(Pb)
        TC = self.rand_array[xn, zn + 1].dot(Pc)
        TD = self.rand_array[xn + 1, zn + 1].dot(Pd)

        l1 = TA + (TB - TA) * xt
        l2 = TC + (TD - TC) * xt
        u = l1 + (l2 - l1) * zt

        z = float(j) * self.dz
        eps = 0.57 * 10.0 ** -2.0
        yita = 2.0 * (z - z0) / B

        vp = u * 2*1000 + v0 * (1.0 + eps * (ti.exp(-yita) - (1.0 - yita)))
        vs = u + v0 * (1.0 + eps * (ti.exp(-yita) - (1.0 - yita)))
        rho = 1000.0 + u

        va = ti.Vector([vp, vs, rho])

        return va

    @ti.kernel
    def model_perlin_munk(self, lx: ti.i32, lz: ti.i32, B: ti.f32, z0: ti.f32, v0: ti.f32):
        for i, j in self.model_vp:
            vdata = self.node(i, j, lx, lz, z0, v0, B)
            self.model_vp[i, j] = vdata[0]
            self.model_vs[i, j] = vdata[1]
            self.model_rho[i, j] = vdata[2]
    
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

    @ti.kernel
    def model_perlin_munk_node(self, lx: ti.i32, lz: ti.i32, B: ti.f32, z0: ti.f32, v0: ti.f32, i:ti.f32, j:ti.f32):
        if i-1<0 or i+1>self.nx*self.dx or j-1<0 or j+1>self.nz*self.dz:
            self.data[0]=1.0
            self.data[1]=0.0
            self.data[2]=0.0
        else:
            vdata = self.node(i, j, lx, lz, z0, v0, B)
            vdata_up = self.node(i, j+1, lx, lz, z0, v0, B)
            vdata_down = self.node(i, j-1, lx, lz, z0, v0, B)
            vdata_left = self.node(i-1, j, lx, lz, z0, v0, B)
            vdata_right = self.node(i+1, j, lx, lz, z0, v0, B)
            vdata_dz = (vdata_up[0] - vdata_down[0])/2
            vdata_dx = (vdata_right[0] - vdata_left[0])/2 
            self.data[0] = vdata[0]
            self.data[1] = vdata_dx
            self.data[2] = vdata_dz



    @ti.kernel
    def model_perlin_ti(self, lx: ti.i32, lz: ti.i32):
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
            TA = self.rand_array[xn, zn].dot(Pa)
            TB = self.rand_array[xn + 1, zn].dot(Pb)
            TC = self.rand_array[xn, zn + 1].dot(Pc)
            TD = self.rand_array[xn + 1, zn + 1].dot(Pd)

            l1 = TA + (TB - TA) * xt
            l2 = TC + (TD - TC) * xt
            u = l1 + (l2 - l1) * zt

            self.model_vp[i, j] = u
            self.model_vs[i, j] = u
            self.model_rho[i, j] = u


    @ti.kernel
    def model_perlin_change(self):
        pass


    def model_perlin(self, lx, lz):
        self.model_perlin_ti(lx, lz)



