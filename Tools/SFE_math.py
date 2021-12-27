import taichi as ti


@ti.data_oriented
class SFE_math():
    def __init__(self):
        self.FD_alpha = 0

    def interp(self, method, IX1, IX2: ti.template(), CX1, CX2, nX1,
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

        for i1 in range(FO[f1][0], FO[f1][1] + 1):
            for i2 in range(FO[f2][0], FO[f2][1] + 1):
                h[i1][i2] = Data[i1 + IX1 - 1, i2 + IX2 - 1]

        if f1 != 2:
            for i2 in range(0, 4):
                h[]
