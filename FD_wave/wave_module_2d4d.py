import taichi as ti
import numpy as np

@ti.data_oriented
class wave:
    def __init__(self, src_x, src_z, nx, nz, dx, dz, PML_up_ToF, PML_n, mod_flag, dt, t, fm):
        self.src_x = src_x
        self.src_z = src_z
        self.nx = nx

        self.nz = nz
        self.t = t
        self.dt = dt
        self.fm = fm
        self.PML_up_ToF = PML_up_ToF
        self.mod_flag = mod_flag
        self.PML_n = PML_n
        self.dx = dx
        self.dz = dz

        self.fn = 2

        self.model = ti.Vector.field(2, dtype=ti.f32, shape=(nx, nz))
        self.p = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.p_field = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx, self.nz))
        self.v_field = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx, self.nz))

        self.C_wave = ti.Vector(
            [9.0 / (8.0 * self.dx), 1.0 / (24.0 * self.dx), 9.0 / (8.0 * self.dz), 1.0 / (24.0 * self.dz),
             1.0 / self.dx ** 3.0, 1.0 / self.dz ** 3.0, 1.0 / self.dx ** 2.0, 1.0 / self.dz ** 2.0,
             self.dt ** 3.0 / 24.0])
        self.p_w = ti.Vector.field(2, dtype=ti.f32, shape=(nx, nz))
        self.v_w = ti.Vector.field(2, dtype=ti.f32, shape=(nx, nz))
        self.d = ti.Vector.field(2, dtype=ti.f32, shape=(nx, nz))


    @ti.func
    def ricker(self, frame):
        return 1.0 * (1 - 2 * (3.1415926 * self.fm * self.dt * frame) ** 2) \
               * ti.exp(-(3.1415926 * self.fm * self.dt * frame) ** 2)


    @ti.kernel
    def mod_default(self):
        for i, j in self.model:
            if j < self.nz / 2:
                self.model[i, j] = ti.Vector([3000.0, 1.0])
            else:
                self.model[i, j] = ti.Vector([2500.0, 1.0])

    def mod_file(self, path):
        arr = np.loadtxt(path)
        self.model = arr.from_numpy()

    @ti.kernel
    def PML_cal(self):
        for i, j in self.d:
            self.d[i, j] = [0, 0]
        
        for i, j in self.d:
            if i <= self.PML_n:
                self.d[i, j][0] = 1.0 * (1.0 - ti.cos(3.1415926 * (float(self.PML_n) - float(i)) / (2.0 * float(self.PML_n))))
                self.d[i, j][1] = 1.0 * (1.0 - ti.cos(3.1415926 * (float(self.PML_n) - float(i)) / (2.0 * float(self.PML_n))))

            if j <= self.PML_n:
                self.d[i, j][0] = 1.0 * (1.0 - ti.cos(3.1415926 * (float(self.PML_n) - float(j)) / (2.0 * float(self.PML_n))))
                self.d[i, j][1] = 1.0 * (1.0 - ti.cos(3.1415926 * (float(self.PML_n) - float(j)) / (2.0 * float(self.PML_n))))

            if self.nx - self.PML_n <= i <= self.nx:
                h = self.nx -i
                self.d[i, j][0] = 1.0 * (1.0 - ti.cos(3.1415926 * (float(self.PML_n) - float(h)) / (2.0 * float(self.PML_n))))
                self.d[i, j][1] = 1.0 * (1.0 - ti.cos(3.1415926 * (float(self.PML_n) - float(h)) / (2.0 * float(self.PML_n))))

            if self.nz - self.PML_n <= j <= self.nz:
                h = self.nz - j
                self.d[i, j][0] = 1.0 * (1.0 - ti.cos(3.1415926 * (float(self.PML_n) - float(h)) / (2.0 * float(self.PML_n))))
                self.d[i, j][1] = 1.0 * (1.0 - ti.cos(3.1415926 * (float(self.PML_n) - float(h)) / (2.0 * float(self.PML_n))))


    @ti.kernel
    def wave_field_cal(self, frame: ti.f32):
        for i, j in self.v_field:
            if self.fn < i < self.nx - self.fn and self.fn < j < self.nz - self.fn:
                self.p_w[i, j][0] = self.C_wave[0] * (self.p[i, j + 1] - self.p[i, j]) - \
                                    self.C_wave[1] * (self.p[i, j + 2] - self.p[i, j - 1])
                self.p_w[i, j][1] = self.C_wave[2] * (self.p[i + 1, j] - self.p[i, j]) - \
                                    self.C_wave[3] * (self.p[i + 2, j] - self.p[i - 1, j])
                self.v_field[i, j][0] = self.v_field[i, j][0] - self.dt / self.model[i, j][1] * self.p_w[i, j][0] - self.d[i, j][0] * self.v_field[i, j][0]
                self.v_field[i, j][1] = self.v_field[i, j][1] - self.dt / self.model[i, j][1] * self.p_w[i, j][1] - self.d[i, j][1] * self.v_field[i, j][1]

        self.p_field[self.src_x, self.src_z][0] = self.ricker(frame)/1.0
        self.p_field[self.src_x, self.src_z][1] = self.ricker(frame)/1.0

        for i, j in self.p_field:
            if self.fn < i < self.nx - self.fn and self.fn < j < self.nz - self.fn:
                self.v_w[i, j][0] = self.C_wave[0] * (self.v_field[i, j][0] - self.v_field[i, j - 1][0]) - \
                                    self.C_wave[1] * (self.v_field[i, j + 1][0] - self.v_field[i, j - 2][0])
                self.v_w[i, j][1] = self.C_wave[2] * (self.v_field[i, j][1] - self.v_field[i - 1, j][1]) - \
                                    self.C_wave[3] * (self.v_field[i + 1, j][1] - self.v_field[i - 2, j][1])
                self.p_field[i, j][0] = self.p_field[i, j][0] - self.model[i, j][0] ** 2 * self.model[i, j][1] * self.dt * (
                        self.v_w[i, j][0]) - self.d[i, j][0] * self.p_field[i, j][0]
                self.p_field[i, j][1] = self.p_field[i, j][1] - self.model[i, j][0] ** 2 * self.model[i, j][1] * self.dt * (
                        self.v_w[i, j][1]) - self.d[i, j][1] * self.p_field[i, j][1]
                self.p[i, j] = self.p_field[i, j][0] + self.p_field[i, j][1]
        

     