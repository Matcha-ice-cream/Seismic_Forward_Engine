import taichi as ti
import numpy as np


@ti.data_oriented
class receiver:
    def __init__(self, mod, rec_n, nt):
        self.mod = mod
        self.rec_n = rec_n
        self.nt = nt
        self.rec_pos_f = ti.Vector.field(2, dtype=ti.f32, shape=rec_n)
        self.rec_pos_f_0 = ti.Vector.field(2, dtype=ti.f32, shape=rec_n)
        self.rec_pos_i = ti.Vector.field(2, dtype=ti.i32, shape=rec_n)
        self.rec_value = ti.field(dtype=ti.f32, shape=(rec_n, nt))

        self.PM = ti.field(dtype=ti.f32, shape=200)

        self.e = ti.field(dtype=ti.f32, shape=200)

    @ti.func
    def weight(self, r, h):
        R = r / h
        a = 15.0 / (7.0 * 3.1415926 * h ** 2.0)
        w1 = 0.0
        if R <= 2.0:
            w1 = a * (2.0 / 3.0 - 9.0 / 8.0 * R ** 2.0 + 19.0 / 24.0 * R ** 3.0 - 5.0 / 32.0 * R ** 4.0)
        else:
            w1 = 0.0
        return w1

    @ti.kernel
    def rec_init(self, nx: ti.i32, nz: ti.i32):
        for i in self.rec_pos_f:
            self.rec_pos_f_0[i][0] = float(nx) / 2.0 + float(i - self.rec_n / 2)*0.2
            self.rec_pos_f_0[i][1] = float(2 * nz / 3)
        for i in self.rec_pos_i:
            self.rec_pos_i[i][0] = nx / 2 + i - self.rec_n / 2
            self.rec_pos_i[i][1] = 2 * nz / 3
        for i, j in self.rec_value:
            self.rec_value[i, j] = 0.0
        for i in self.e:
            self.e[i] = ti.random() * 2.0 * 3.1415926

    def rec_from_file(self, path):
        arr = np.loadtxt(path)
        self.rec_pos_f_0 = arr.from_numpy()
        self.rec_pos_i = arr.from_numpy()

        

    @ti.kernel
    def rec_dynamic(self, dt: ti.f32, frame: ti.i32, wind_v: ti.f32):
        for i in self.PM:
            self.PM[i] = (8.1 * 10.0 ** (-3.0) * 9.8 ** 2.0) / ((float(i+1) * 3.1415926 / 100.0) ** 5.0) * \
                         ti.exp(-0.74 * (9.8 / (wind_v * float(i+1) * 3.1415926 / 100.0)) ** 4.0)

        for i in self.rec_pos_f:
            self.rec_pos_f[i] = [0.0, 0.0]
            for j in range(200):
                w = float(j) * 3.1415926 / 100.0
                t = dt * frame
                self.rec_pos_f[i][0] = self.rec_pos_f[i][0] + self.PM[j] * ti.sin(
                    w ** 2.0 / 9.8 * self.rec_pos_f_0[i][0] - w * t + self.e[j])
                self.rec_pos_f[i][1] = self.rec_pos_f[i][1] + self.PM[j] * ti.cos(
                    w ** 2.0 / 9.8 * self.rec_pos_f_0[i][0] - w * t + self.e[j])
            self.rec_pos_f[i][0] = self.rec_pos_f_0[i][0] - self.rec_pos_f[i][0]
            self.rec_pos_f[i][1] = self.rec_pos_f_0[i][1] + self.rec_pos_f[i][1]

    @ti.kernel
    def rec_gather(self, wave: ti.template(), t: ti.i32):
        if self.mod == 'node':
            for i in self.rec_pos_i:
                self.rec_value[i, self.nt - t + 1] = wave[(self.rec_pos_i[i][0]), (self.rec_pos_i[i][1])]

        if self.mod == 'PIC':
            for i in self.rec_pos_f:
                center = ti.Vector([int(self.rec_pos_f[i][0] - 0.5), int(self.rec_pos_f[i][1] - 0.5)])
                for j in range(3):
                    for k in range(3):
                        xy = ti.Vector([center[0] + j - 1, center[1] + k - 1])
                        r = ((xy[0] - self.rec_pos_f[i][0]) ** 2.0 + (xy[1] - self.rec_pos_f[i][1]) ** 2.0) ** 0.5
                        w = self.weight(r, 3.0)
                        self.rec_value[i, self.nt - t + 1] = self.rec_value[i, self.nt - t + 1] + w * wave[xy[0], xy[1]]

    def export(self, arr, path):
        arr_export = arr.to_numpy()
        np.savetxt(path,arr_export)




