import taichi as ti


@ti.data_oriented
class receiver:
    def __init__(self, mod, rec_n, nt):
        self.mod = mod
        self.rec_n = rec_n
        self.nt = nt
        self.rec_pos_f = ti.Vector.field(2, dtype=ti.f32, shape=rec_n)
        self.rec_pos_i = ti.Vector.field(2, dtype=ti.i32, shape=rec_n)
        self.rec_value = ti.field(dtype=ti.f32, shape=(rec_n, nt))

        self.PM = ti.field(dtype=ti.f32, shape=200)

    @ti.kernel
    def rec_init(self, nx: ti.i32, nz: ti.i32):
        for i in self.rec_pos_f:
            self.rec_pos_f[i][0] = float(nx) / 2.0 + float(i - self.rec_n / 2)
            self.rec_pos_f[i][1] = float(2 * nz / 3)
        for i in self.rec_pos_i:
            self.rec_pos_i[i][0] = nx / 2 + i - self.rec_n / 2
            self.rec_pos_i[i][1] = 2 * nz / 3
        for i, j in self.rec_value:
            self.rec_value[i, j] = 0.0

    @ti.kernel
    def rec_dynamic(self, dt: ti.f32, frame: ti.i32, wind_v: ti.f32):
        for i in self.PM:
            self.PM[i] = (8.1 * 10.0 ** (-3.0) * 9.8 ** 2.0) / ((float(i) * 3.1415926 / 100.0) ** 5.0) * \
                         ti.exp(-0.74 * (9.8 / (wind_v * float(i) * 3.1415926 / 100.0)) ** 4.0)

        for i in self.rec_pos_f:
            for j in range(200):
                w = float(j) * 3.1415926 / 100.0
                t = dt * frame
                self.rec_pos_f[i][0] = self.rec_pos_f[i][0] + self.PM[i] * ti.sin(
                    w ** 2.0 / 9.8 * self.rec_pos_f[i][0] + w * t)
                self.rec_pos_f[i][1] = self.rec_pos_f[i][1] + self.PM[i] * ti.cos(
                    w ** 2.0 / 9.8 * self.rec_pos_f[i][0] + w * t)

    @ti.kernel
    def rec_gather(self, wave: ti.template(), t: ti.i32):
        if self.mod == 'node':
            for i in self.rec_pos_i:
                self.rec_value[i, 1000 - t + 1] = wave[(self.rec_pos_i[i][0]), (self.rec_pos_i[i][1])]

        if self.mod == 'PIC':
            for i in self.rec_pos_f:
                center = ti.Vector([int(self.rec_pos_f[i][0] - 0.5), int(self.rec_pos_f[i][1] - 0.5)])
                for j in range(3):
                    for k in range(3):
                        xy = ti.Vector([center[0] + j - 1, center[1] + k - 1])
                        r = ((xy[0] - self.rec_pos_f[i][0]) ** 2.0 + (xy[1] - self.rec_pos_f[i][1]) ** 2.0) ** 0.5
                        w = (15.0 / (3.1415926 * 2.0 ** 6.0)) * (2.0 - r) ** 3.0
                        self.rec_value[i, 1000 - t + 1] = self.rec_value[i, 1000 - t + 1] + w * wave[xy[0], xy[1]]
