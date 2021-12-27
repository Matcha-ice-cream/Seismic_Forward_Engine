import taichi as ti


@ti.data_oriented
class SFE_math():
    def __init__(self):
        self.FD_alpha = 0

    def interp(self, method, IX1, IX2: ti.template(), CX1, CX2, nX1,
               nX2, Data, Re, ReX1, ReX2):
        self.h = ti.field(ti.f32, shape=(4, 4))
        self.FX1 = ti.field(ti.f32, shape=4)
        self.FX2 = ti.field(ti.f32, shape=4)
        self.DX1 = ti.field(ti.f32, shape=4)
        self.DX2 = ti.field(ti.f32, shape=4)
        self.TE = ti.field(ti.f32, shape=4)

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

        self.FX1[0] = -CX1th + 2 * CX1tw - CX1
        self.FX1[1] = 3 * CX1th - 5 * CX1tw
        self.FX1[2] = -3 * CX1th + 4 * CX1tw + CX1
        self.FX1[3] = CX1th - CX1tw

        self.FX2[0] = -CX2th + 2 * CX2tw - CX2
        self.FX2[1] = 3 * CX2th - 5 * CX2tw
        self.FX2[2] = -3 * CX2th + 4 * CX2tw + CX2
        self.FX2[3] = CX2th - CX2tw

        self.DX1[0] = -3 * CX1tw + 4 * CX1 - 1
        self.DX1[1] = 9 * CX1tw - 10 * CX1
        self.DX1[2] = -9 * CX1tw + 8 * CX1 + 1
        self.DX1[3] = 3 * CX1tw - 2 * CX1

        self.DX2[0] = -3 * CX2tw + 4 * CX2 - 1
        self.DX2[1] = 9 * CX2tw - 10 * CX2
        self.DX2[2] = -9 * CX2tw + 8 * CX2 + 1
        self.DX2[3] = 3 * CX2tw - 2 * CX2

        FO = ti.Vector([[1, 3], [0, 2], [0, 3]])



