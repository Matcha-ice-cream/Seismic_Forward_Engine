import taichi as ti

@ti.data_oriented
class receiver:
    def __init__(self, mod, rec_n, nt):
        self.mod = mod
        self.rec_n = rec_n
        self.nt = nt
        self.rec_pos = ti.Vector.field(2, dtype=ti.f32, shape=rec_n)
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
        for i in self.rec_pos:
            self.rec_pos[i][0] = nx / 2 + float(i - self.rec_n/2)
            self.rec_pos[i][1] = float(2 * nz / 3)

    @ti.kernel
    def rec_dynamic(self):
        pass

    @ti.kernel
    def rec_gather(self, nx: ti.i32, nz: ti.i32, wave: ti.template()):
        if self.mod == 1:
            self.mod_int()
        
        for i in self.rec_pos:
            self.rec_value = wave[int(self.rec_pos[i][0]), int(self.rec_pos[i][1])]



