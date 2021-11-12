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

    
    @ti.func
    def mod_int(self):
        for i in self.rec_pos:
            self.rec_pos[i][0] = int(self.rec_pos[i][0])
            self.rec_pos[i][1] = int(self.rec_pos[i][0])


    @ti.func
    def mod_PIC(self):
        
        pass

    @ti.kernel
    def rec_default(self, nx:ti.i32, nz:ti.i32):
        for i in self.rec_pos_f:
            self.rec_pos_f[i][0] = float(nx) / 2.0 + float(i - self.rec_n/2)
            self.rec_pos_f[i][1] = float(2 * nz / 3)
        for i in self.rec_pos_i:
            self.rec_pos_i[i][0] = nx / 2 + i - self.rec_n/2
            self.rec_pos_i[i][1] = 2 * nz / 3


    @ti.kernel
    def rec_dynamic(self):
        pass

    @ti.kernel
    def rec_gather(self, nx: ti.i32, nz: ti.i32, wave: ti.template(), t:ti.i32):
        if self.mod == 'node':
            for i in self.rec_pos_i:
                self.rec_value[i, t] = wave[(self.rec_pos_i[i][0]), (self.rec_pos_i[i][1])]

        if self.mod == 'dynamic':
            for i in self.rec_pos_f:
                pass



