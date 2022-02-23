import taichi as ti
import sys
sys.path.append("..")
from model.model_operation import getmodel



@ti.data_oriented
class RK_ray:
    def __init__(self, src_x, src_z, dt, frame, n):
        self.src_x = src_x
        self.src_z = src_z
        self.dt = dt
        self.n = n
        self.frame = frame

        self.ray_path = ti.Vector.field(2, dtype=ti.f32, shape=(n, frame))
        self.ray_direction = ti.Vector.field(2, dtype=ti.f32, shape=(n, frame))
        self.ray_position = ti.Vector.field(2, dtype=ti.f32, shape=(frame))

        self.a = ti.Vector.field(2, dtype=ti.f32, shape=1)
        self.b = ti.Vector.field(2, dtype=ti.f32, shape=1)

        self.rand_array = ti.Vector.field(2, dtype=ti.f32, shape=(100, 100))


    @ti.kernel
    def RK_ray_init(self):
        for i in range(self.n):
            cos_x = ti.cos(float(i) / float(self.n) * 3.1415926)
            sin_x = ti.sin(float(i) / float(self.n) * 3.1415926)
            self.ray_direction[i, 0][0] = cos_x
            self.ray_direction[i, 0][1] = sin_x

            self.ray_path[i, 0][0] = self.src_x
            self.ray_path[i, 0][1] = self.src_z
        for i, j in self.rand_array:
            a = (ti.random(ti.f32) - 0.5) * 2
            b = (ti.random(ti.f32) - 0.5) * 2
            self.rand_array[i, j] = ti.Vector([a, b]).normalized()

    @ti.func
    def fun(self, data0, data):
        Y2 = ti.Vector([data0[2], data0[3], -1/(data[0]**3)*data[1], -1/(data[0]**3)*data[2]]) 
        return Y2

    @ti.func
    def fade(self, t):
        return 6.0 * t ** 5.0 - 15.0 * t ** 4.0 + 10.0 * t ** 3.0

    @ti.func
    def node(self, i, j, lx, lz, z0, v0, B, dz):
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

        z = float(j) * dz
        eps = 0.57 * 10.0 ** -2.0
        yita = 2.0 * (z - z0) / B

        vp = u * 2*1000 + v0 * (1.0 + eps * (ti.exp(-yita) - (1.0 - yita)))
        vs = u + v0 * (1.0 + eps * (ti.exp(-yita) - (1.0 - yita)))
        rho = 1000.0 + u

        va = ti.Vector([vp, vs, rho])

        return va

    @ti.func
    def node_diff(self, lx, lz, B, z0, v0, i, j, nx, nz, dx, dz):
        data = ti.Vector([0.0,0.0,0.0])
        if i-1<0 or i+1>nx*dx or j-1<0 or j+1>nz*dz:
            data[0]=1.0
            data[1]=0.0
            data[2]=0.0
        else:
            vdata = self.node(i, j, lx, lz, z0, v0, B, dz)
            vdata_up = self.node(i, j+1, lx, lz, z0, v0, B, dz)
            vdata_down = self.node(i, j-1, lx, lz, z0, v0, B, dz)
            vdata_left = self.node(i-1, j, lx, lz, z0, v0, B, dz)
            vdata_right = self.node(i+1, j, lx, lz, z0, v0, B, dz)
            vdata_dz = (vdata_up[0] - vdata_down[0])/2
            vdata_dx = (vdata_right[0] - vdata_left[0])/2 
            data[0] = vdata[0]
            data[1] = vdata_dx
            data[2] = vdata_dz
        return data


    
    @ti.func
    def ray_trace(self, data0, data, lx, lz, B, z0, v0, nx, nz, dx, dz):
        pos = ti.Vector([0.0,0.0,0.0,0.0])
        K1 =  self.fun(data0, data)
        pos = data0+self.dt*K1/2
        data1 = self.node_diff(lx, lz, B, z0, v0, pos[0], pos[1], nx, nz, dx, dz)
        K2 = self.fun(data0+self.dt*K1/2, data1)
        pos = data0+self.dt*K2/2
        data2 = self.node_diff(lx, lz, B, z0, v0, pos[0], pos[1], nx, nz, dx, dz)
        K3 = self.fun(data0 + self.dt*K2/2, data2)
        pos = data0+self.dt*K3
        data3 = self.node_diff(lx, lz, B, z0, v0, pos[0], pos[1], nx, nz, dx, dz)
        K4 = self.fun(data0+self.dt*K3, data3)
        data4 = data0 + self.dt/6*(K1 + 2*K2+2*K3+K4)
        return data4


    @ti.kernel
    def RK_ray_substep(self, k:ti.i32, data:ti.template(), lx:ti.f32, lz:ti.f32, B:ti.f32, z0:ti.f32, v0:ti.f32, nx:ti.i32, nz:ti.i32, dx:ti.f32, dz:ti.f32):
        for i in range(self.n):
            data0 = ti.Vector([self.ray_path[i, k][0], self.ray_path[i, k][1], self.ray_direction[i, k][0], self.ray_direction[i, k][1]])
            # data1 = ti.Vector([self.ray_path[i, k+1][0], self.ray_path[i, k+1][1], self.ray_direction[i, k+1][0], self.ray_direction[i, k+1][1]])
            data1 = self.ray_trace(data0, data, lx, lz, B, z0, v0, nx, nz, dx, dz)
            self.ray_path[i, k+1][0] = data1[0]
            self.ray_path[i, k+1][1] = data1[1]
            self.ray_direction[i, k+1][0] = data1[2]
            self.ray_direction[i, k+1][1] = data1[3]
            if self.ray_path[i, k+1][0] == 0.0 or self.ray_path[i, k+1][1] == 0.0:
                self.ray_path[i, k+1][0] = self.ray_path[i, k][0]
                self.ray_path[i, k+1][1] = self.ray_path[i, k][1]




    @ti.kernel
    def RK_ray_data_tidy(self, xn:ti.i32, nx:ti.f32, nz:ti.f32):
        for i in self.ray_position:
            aa = ti.Vector([self.ray_path[xn, i][0]/nx,1 - self.ray_path[xn, i][1]/nz])
            self.ray_position[i] = aa
    
    def RK_ray_paint_single(self, gui, xn, nx, nz, color, radius):
        self.RK_ray_data_tidy(xn, nx, nz)
        print(self.ray_position)

        while True:
            gui.circles(self.ray_position.to_numpy(),color=color, radius=radius)
            gui.show()

    def RK_ray_paint_multi(self, gui, nx, nz, color, radius):
        while True:
            for i in range(self.n-1):
                self.RK_ray_data_tidy(i+1, nx, nz)
                gui.circles(self.ray_position.to_numpy(), color=color, radius=radius)
            gui.show()
            # print(self.ray_position)


    @ti.func
    def dis(self, x1, y1, x2, y2):
        return ((x1 - x2)**2.0 + (y1 - y2)**2.0) ** 0.5

    @ti.func
    def interp(self, IX1, IX2, CX1, CX2, nX1,
               nX2, Data, Re, ReX1, ReX2):
        h = ti.field(ti.f32, shape=(4, 4))
        FX1 = ti.field(ti.f32, shape=4)
        FX2 = ti.field(ti.f32, shape=4)
        DX1 = ti.field(ti.f32, shape=4)
        DX2 = ti.field(ti.f32, shape=4)
        TE = ti.field(ti.f32, shape=4)

        if IX1 == nX1 - 1:
            IX1 = IX1 - 1
            CX1 = 1

        if IX2 == nX2 - 1:
            IX2 = IX2 - 1
            CX2 = 1

        CX1tw = CX1 ** 2.0
        CX1th = CX1 * CX1tw
        CX2tw = CX2 ** 2.0
        CX2th = CX2 * CX2tw

        FX1[0] = -CX1th + 2 * CX1tw - CX1
        FX1[1] = 3 * CX1th - 5 * CX1tw
        FX1[2] = -3 * CX1th + 4 * CX1tw + CX1
        FX1[3] = CX1th - CX1tw

        FX2[0] = -CX2th + 2 * CX2tw - CX2
        FX2[1] = 3 * CX2th - 5 * CX2tw
        FX2[2] = -3 * CX2th + 4 * CX2tw + CX2
        FX2[3] = CX2th - CX2tw

        DX1[0] = -3 * CX1tw + 4 * CX1 - 1
        DX1[1] = 9 * CX1tw - 10 * CX1
        DX1[2] = -9 * CX1tw + 8 * CX1 + 1
        DX1[3] = 3 * CX1tw - 2 * CX1

        DX2[0] = -3 * CX2tw + 4 * CX2 - 1
        DX2[1] = 9 * CX2tw - 10 * CX2
        DX2[2] = -9 * CX2tw + 8 * CX2 + 1
        DX2[3] = 3 * CX2tw - 2 * CX2

        FO = ti.Vector([[1, 3], [0, 2], [0, 3]])
        ID = ti.Vector([[0, 1, 2, 3], [3, 2, 1, 0]])

        if IX1 == 0:
            f1 = 0
        elif IX1 == nX1 - 2:
            f1 = 1
        else:
            f1 = 2

        if IX2 == 0:
            f2 = 0
        elif IX2 == nX2 - 2:
            f2 = 1
        else:
            f2 = 2

        for i1 in range(FO[f1, 0], FO[f1, 1] + 1):
            for i2 in range(FO[f2, 0], FO[f2, 1] + 1):
                h[i1, i2] = Data[i1 + IX1 - 1, i2 + IX2 - 1]

        if f1 != 2:
            for i2 in range(0, 4):
                h[ID[f1, 0], i2] = 3 * (h[ID[f1, 1], i2] - h[ID[f1, 2], i2]) + h[ID[f1, 3], i2]

        if f2 != 2:
            for i1 in range(0, 4):
                h[i1, ID[f2, 0]] = 3 * (h[i1, ID[f2, 1]] - h[i1, ID[f2, 2]]) + h[i1, ID[f2, 3]]

        for i2 in range(0, 4):
            TE[i2] = 0
            for i1 in range(0, 4):
                TE[i2] = TE[i2] + FX1[i1] * h[i1, i2]

        Re = 0
        for i1 in range(0, 4):
            Re = Re + TE[i1] * FX2[i1]
        Re = Re / 4

        ReX2 = 0
        for i1 in range(0, 4):
            ReX2 = ReX2 + TE[i1] * DX2[i1]
        ReX2 = ReX2 / 4

        for i2 in range(0, 4):
            TE[i2] = 0
            for i1 in range(0, 4):
                TE[i2] = TE[i2] + DX1[i1] * h[i1, i2]
        ReX1 = 0
        for i1 in range(0, 4):
            ReX1 = ReX1 + TE[i1] * FX2[i1]

        ReX1 = ReX1 / 4




















