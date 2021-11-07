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

        self.fn = 1

        self.model = ti.Vector.field(2, dtype=ti.f32, shape=(nx, nz))
        self.p_field = ti.field(dtype=ti.f32, shape=(self.nx, self.nz))
        self.v_field = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx, self.nz))

        self.C_wave = ti.Vector(
            [9.0 / (8.0 * self.dx), 1.0 / (24.0 * self.dx), 9.0 / (8.0 * self.dz), 1.0 / (24.0 * self.dz),
             1.0 / self.dx ** 3.0, 1.0 / self.dz ** 3.0, 1.0 / self.dx ** 2.0, 1.0 / self.dz ** 2.0,
             self.dt ** 3.0 / 24.0])
        self.p_w = ti.Vector.field(2, dtype=ti.f32, shape=(nx, nz))
        self.v_w = ti.Vector.field(2, dtype=ti.f32, shape=(nx, nz))

        self.d_x = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.K_x = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.alpha_prime_x = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.a_x = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.b_x = ti.field(dtype=ti.f32, shape=(2 * PML_n))

        self.d_x_half = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.K_x_half = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.alpha_prime_x_half = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.a_x_half = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.b_x_half = ti.field(dtype=ti.f32, shape=(2 * PML_n))

        self.d_z = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.K_z = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.alpha_prime_z = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.a_z = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.b_z = ti.field(dtype=ti.f32, shape=(2 * PML_n))

        self.d_z_half = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.K_z_half = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.alpha_prime_z_half = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.a_z_half = ti.field(dtype=ti.f32, shape=(2 * PML_n))
        self.b_z_half = ti.field(dtype=ti.f32, shape=(2 * PML_n))

        self.psi_p_x = ti.field(dtype=ti.f32, shape=(2 * self.PML_n, self.nz))
        self.psi_p_z = ti.field(dtype=ti.f32, shape=(self.nx, 2 * self.PML_n))
        self.psi_vxx = ti.field(dtype=ti.f32, shape=(2 * self.PML_n, self.nz))
        self.psi_vxxs = ti.field(dtype=ti.f32, shape=(2 * self.PML_n, self.nz))
        self.psi_vzz = ti.field(dtype=ti.f32, shape=(self.nx, 2 * self.PML_n))

    @ti.func
    def ricker(self, frame):
        return 10 * (1 - 2 * (3.1415926 * self.fm * self.dt * frame) ** 2) \
               * ti.exp(-(3.1415926 * self.fm * self.dt * frame) ** 2)

    @ti.kernel
    def PML_pro(self, PML_n: ti.i32, FPML: ti.f32, npower: ti.f32, DAMPING: ti.f32, k_max_PML: ti.f32):
        # PML parameters

        for i, j in self.psi_p_x:
            self.psi_p_x[i, j] = 0.0
            self.psi_vxx[i, j] = 0.0
            self.psi_vxxs[i, j] = 0.0

        for i, j in self.psi_p_z:
            self.psi_p_z[i, j] = 0.0
            self.psi_vzz[i, j] = 0.0

        alpha_max_PML = 2.0 * 3.1415926 * (FPML / 2.0)
        thickness_PML_x = self.PML_n * self.dx
        thickness_PML_z = self.PML_n * self.dz
        Rcoef = 0.001
        NXG = self.nx
        NZG = self.nz
        d0_x = - (npower + 1) * DAMPING * ti.log(Rcoef) / (2.0 * thickness_PML_x)
        d0_z = - (npower + 1) * DAMPING * ti.log(Rcoef) / (2.0 * thickness_PML_z)

        # left & right
        xoriginleft = thickness_PML_x
        xoriginright = (NXG - 1) * self.dx - thickness_PML_x

        for i in range(1, PML_n + 1):
            self.K_x[i] = 1.0
            self.K_x_half[i] = 1.0
            xval = self.dx * (i - 1)

            abscissa_in_PML = xoriginleft - xval

            if abscissa_in_PML >= 0.0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                self.d_x[i] = d0_x * abscissa_normalized ** npower
                self.K_x[i] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized ** npower
                self.alpha_prime_x[i] = alpha_max_PML * (1.0 - abscissa_normalized)

            abscissa_in_PML = xoriginleft - (xval + self.dx / 2.0)
            if abscissa_in_PML >= 0.0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                self.d_x_half[i] = d0_x * abscissa_normalized ** npower
                self.K_x_half[i] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized ** npower
                self.alpha_prime_x_half[i] = alpha_max_PML * (1.0 - abscissa_normalized)

            if self.alpha_prime_x[i] < 0.0:
                self.alpha_prime_x[i] = 0.0
            if self.alpha_prime_x_half[i] < 0.0:
                self.alpha_prime_x_half[i] = 0.0

            self.b_x[i] = ti.exp(- (self.d_x[i] / self.K_x[i] + self.alpha_prime_x[i]) * self.dt)
            self.b_x_half[i] = ti.exp(- (self.d_x_half[i] / self.K_x_half[i] + self.alpha_prime_x_half[i]) * self.dt)

            if ti.abs(self.d_x[i]) > 1.0e-6:
                self.a_x[i] = self.d_x[i] * (self.b_x[i] - 1.0) / (
                        self.K_x[i] * (self.d_x[i] + self.K_x[i] * self.alpha_prime_x[i]))
            if ti.abs(self.d_x_half[i]) > 1.0e-6:
                self.a_x_half[i] = self.d_x_half[i] * (self.b_x_half[i] - 1.0) / (
                        self.K_x_half[i] * (self.d_x_half[i] + self.K_x_half[i] * self.alpha_prime_x_half[i]))

        for i in range(NXG - PML_n + 1, NXG + 1):
            h = i - NXG + 2 * PML_n
            self.K_x[h] = 1.0
            self.K_x_half[h] = 1.0
            xval = self.dx * (i - 1)

            abscissa_in_PML = xval - xoriginright

            if abscissa_in_PML >= 0.0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                self.d_x[h] = d0_x * abscissa_normalized ** npower
                self.K_x[h] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized ** npower
                self.alpha_prime_x[h] = alpha_max_PML * (1.0 - abscissa_normalized)

            abscissa_in_PML = (xval + self.dx / 2.0) - xoriginright
            if abscissa_in_PML >= 0.0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                self.d_x_half[h] = d0_x * abscissa_normalized ** npower
                self.K_x_half[h] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized ** npower
                self.alpha_prime_x_half[h] = alpha_max_PML * (1.0 - abscissa_normalized)

            if self.alpha_prime_x[h] < 0.0:
                self.alpha_prime_x[h] = 0.0
            if self.alpha_prime_x_half[h] < 0.0:
                self.alpha_prime_x_half[h] = 0.0

            self.b_x[h] = ti.exp(- (self.d_x[h] / self.K_x[h] + self.alpha_prime_x[h]) * self.dt)
            self.b_x_half[h] = ti.exp(- (self.d_x_half[h] / self.K_x_half[h] + self.alpha_prime_x_half[h]) * self.dt)

            if ti.abs(self.d_x[h]) > 1.0e-6:
                self.a_x[h] = self.d_x[h] * (self.b_x[h] - 1.0) / (
                        self.K_x[h] * (self.d_x[h] + self.K_x[h] * self.alpha_prime_x[h]))
            if ti.abs(self.d_x_half[i]) > 1.0e-6:
                self.a_x_half[h] = self.d_x_half[h] * (self.b_x_half[h] - 1.0) / (
                        self.K_x_half[h] * (self.d_x_half[h] + self.K_x_half[h] * self.alpha_prime_x_half[h]))

        # top & bottom
        zoriginbottom = (NZG - 1) * self.dz - thickness_PML_z
        zorigintop = thickness_PML_z
        for i in range(1, PML_n + 1):
            self.K_z[i] = 1.0
            self.K_z_half[i] = 1.0
            zval = self.dz * (i - 1)

            abscissa_in_PML = zoriginbottom - zval

            if abscissa_in_PML >= 0.0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_z
                self.d_z[i] = d0_x * abscissa_normalized ** npower
                self.K_z[i] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized ** npower
                self.alpha_prime_z[i] = alpha_max_PML * (1.0 - abscissa_normalized)

            abscissa_in_PML = zoriginbottom - (zval + self.dz / 2.0)
            if abscissa_in_PML >= 0.0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_z
                self.d_z_half[i] = d0_z * abscissa_normalized ** npower
                self.K_z_half[i] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized ** npower
                self.alpha_prime_z_half[i] = alpha_max_PML * (1.0 - abscissa_normalized)

            if self.alpha_prime_z[i] < 0.0:
                self.alpha_prime_z[i] = 0.0
            if self.alpha_prime_z_half[i] < 0.0:
                self.alpha_prime_z_half[i] = 0.0

            self.b_z[i] = ti.exp(- (self.d_z[i] / self.K_z[i] + self.alpha_prime_z[i]) * self.dt)
            self.b_z_half[i] = ti.exp(- (self.d_z_half[i] / self.K_z_half[i] + self.alpha_prime_z_half[i]) * self.dt)

            if ti.abs(self.d_z[i]) > 1.0e-6:
                self.a_z[i] = self.d_z[i] * (self.b_z[i] - 1.0) / (
                        self.K_z[i] * (self.d_z[i] + self.K_z[i] * self.alpha_prime_z[i]))
            if ti.abs(self.d_z_half[i]) > 1.0e-6:
                self.a_z_half[i] = self.d_z_half[i] * (self.b_z_half[i] - 1.0) / (
                        self.K_z_half[i] * (self.d_z_half[i] + self.K_z_half[i] * self.alpha_prime_z_half[i]))

        for i in range(NZG - PML_n + 1, NZG + 1):
            h = i - NZG + 2 * PML_n
            self.K_z[h] = 1.0
            self.K_z_half[h] = 1.0
            zval = self.dz * (i - 1)

            abscissa_in_PML = zval - zorigintop

            if abscissa_in_PML >= 0.0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_z
                self.d_z[h] = d0_z * abscissa_normalized ** npower
                self.K_z[h] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized ** npower
                self.alpha_prime_z[h] = alpha_max_PML * (1.0 - abscissa_normalized)

            abscissa_in_PML = (zval + self.dz / 2.0) - zorigintop
            if abscissa_in_PML >= 0.0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_z
                self.d_z_half[h] = d0_x * abscissa_normalized ** npower
                self.K_z_half[h] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized ** npower
                self.alpha_prime_z_half[h] = alpha_max_PML * (1.0 - abscissa_normalized)

            if self.alpha_prime_z[h] < 0.0:
                self.alpha_prime_z[h] = 0.0
            if self.alpha_prime_z_half[h] < 0.0:
                self.alpha_prime_z_half[h] = 0.0

            self.b_z[h] = ti.exp(- (self.d_z[h] / self.K_z[h] + self.alpha_prime_z[h]) * self.dt)
            self.b_z_half[h] = ti.exp(- (self.d_z_half[h] / self.K_z_half[h] + self.alpha_prime_z_half[h]) * self.dt)

            if ti.abs(self.d_z[h]) > 1.0e-6:
                self.a_z[h] = self.d_z[h] * (self.b_z[h] - 1.0) / (
                        self.K_z[h] * (self.d_z[h] + self.K_z[h] * self.alpha_prime_z[h]))
            if ti.abs(self.d_z_half[i]) > 1.0e-6:
                self.a_z_half[h] = self.d_z_half[h] * (self.b_z_half[h] - 1.0) / (
                        self.K_z_half[h] * (self.d_z_half[h] + self.K_z_half[h] * self.alpha_prime_z_half[h]))

    @ti.func
    def PML_v(self, i: ti.i32, j: ti.i32):
        if i <= self.PML_n:
            self.psi_vxx[i, j] = self.b_x[i] * self.psi_vxx[i, j] + self.a_x[i] * self.v_w[i, j][0]
            self.v_w[i, j][0] = self.v_w[i, j][0] / self.K_x[i] + self.psi_vxx[i, j]

        if i >= self.nx - self.PML_n + 1:
            h1 = (i - self.nx + 2 * self.PML_n)
            h = i
            self.psi_vxx[h1, j] = self.b_x[h1] * self.psi_vxx[h1, j] + self.a_x[h1] * self.v_w[i, j][0]
            self.v_w[i, j][0] = self.v_w[i, j][0] / self.K_x[h1] + self.psi_vxx[h1, j]

        if j <= self.PML_n:
            self.psi_vzz[i, j] = self.b_z[j] * self.psi_vzz[i, j] + self.a_z[j] * self.v_w[i, j][1]
            self.v_w[i, j][1] = self.v_w[i, j][1] / self.K_z[j] + self.psi_vzz[i, j]

        if j >= self.nz - self.PML_n + 1:
            h1 = (j - self.nz + 2 * self.PML_n)
            h = j
            self.psi_vzz[i, h1] = self.b_z[h1] * self.psi_vzz[i, h1] + self.a_z[h1] * self.v_w[i, j][1]
            self.v_w[i, j][1] = self.v_w[i, j][1] / self.K_z[h1] + self.psi_vzz[i, h1]

    @ti.func
    def PML_p(self, i: ti.i32, j: ti.i32):
        if i <= self.PML_n:
            self.psi_p_x[i, j] = self.b_x_half[i] * self.psi_p_x[i, j] + self.a_x_half[i] * self.p_w[i, j][0]
            self.p_w[i, j][0] = self.p_w[i, j][0] / self.K_x_half[i] + self.psi_p_x[i, j]

        if i >= self.nx - self.PML_n + 1:
            h1 = (i - self.nx + 2 * self.PML_n)
            h = i
            self.psi_p_x[h1, j] = self.b_x_half[h1] * self.psi_p_x[h1, j] + self.a_x_half[h1] * self.p_w[i, j][0]
            self.p_w[i, j][0] = self.p_w[i, j][0] / self.K_x_half[h1] + self.psi_p_x[h1, j]

        if j <= self.PML_n:
            self.psi_p_z[i, j] = self.b_z_half[j] * self.psi_p_z[i, j] + self.a_z_half[j] * self.p_w[i, j][1]
            self.p_w[i, j][1] = self.p_w[i, j][1] / self.K_z_half[j] + self.psi_p_z[i, j]

        if j >= self.nz - self.PML_n + 1:
            h1 = (j - self.nz + 2 * self.PML_n)
            h = j
            self.psi_p_z[i, h1] = self.b_z_half[h1] * self.psi_p_z[i, h1] + self.a_z_half[h1] * self.p_w[i, j][1]
            self.p_w[i, j][1] = self.p_w[i, j][1] / self.K_z_half[h1] + self.psi_p_z[i, h1]

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
    def wave_field_cal(self, frame: ti.f32):
        for i, j in self.v_field:
            if self.PML_n <= i <= self.nx - self.PML_n and self.PML_n <= j <= self.nz - self.PML_n:
                self.p_w[i, j][0] = self.C_wave[0] * (self.p_field[i, j + 1] - self.p_field[i, j]) - \
                                    self.C_wave[1] * (self.p_field[i, j + 2] - self.p_field[i, j - 1])
                self.p_w[i, j][1] = self.C_wave[2] * (self.p_field[i + 1, j] - self.p_field[i, j]) - \
                                    self.C_wave[3] * (self.p_field[i + 2, j] - self.p_field[i - 1, j])

                self.v_field[i, j][0] = self.v_field[i, j][0] - self.dt / self.model[i, j][1] * self.p_w[i, j][0]
                self.v_field[i, j][1] = self.v_field[i, j][1] - self.dt / self.model[i, j][1] * self.p_w[i, j][1]

            else:
                pass

        self.p_field[self.src_x, self.src_z] = self.ricker(frame)

        for i, j in self.p_field:
            if self.PML_n <= i <= self.nx - self.PML_n and self.PML_n <= j <= self.nz - self.PML_n:
                self.v_w[i, j][0] = self.C_wave[0] * (self.v_field[i, j][0] - self.v_field[i, j - 1][0]) - \
                                    self.C_wave[1] * (self.v_field[i, j + 1][0] - self.v_field[i, j - 2][0])
                self.v_w[i, j][1] = self.C_wave[2] * (self.v_field[i, j][1] - self.v_field[i - 1, j][1]) - \
                                    self.C_wave[3] * (self.v_field[i + 1, j][1] - self.v_field[i - 2, j][1])

                self.p_field[i, j] = self.p_field[i, j] - self.model[i, j][0] ** 2 * self.model[i, j][1] * self.dt * (
                        self.v_w[i, j][0] + self.v_w[i, j][1])
            else:
                pass
