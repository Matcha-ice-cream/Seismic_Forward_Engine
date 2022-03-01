from importlib.machinery import SOURCE_SUFFIXES
import taichi as ti

@ti.data_oriented
class ray:
    def __init__(self, src_x, src_z, dt, frame, n, dx, dz):
        self.src_x = src_x
        self.src_z = src_z
        self.dt = dt
        self.n = n
        self.frame = frame
        self.dx = dx
        self.dz = dz

        self.ray_path = ti.Vector.field(2, dtype=ti.f32, shape=(n, frame))
        self.ray_direction = ti.Vector.field(2, dtype=ti.f32, shape=(n, frame))
        self.ray_position = ti.Vector.field(2, dtype=ti.f32, shape=(frame))

        self.a = ti.Vector.field(2, dtype=ti.f32, shape=1)
        self.b = ti.Vector.field(2, dtype=ti.f32, shape=1)


    @ti.kernel
    def ray_init(self):
        for i in range(self.n):
            cos_x = ti.cos(float(i) / float(self.n) * 3.1415926)
            sin_x = ti.sin(float(i) / float(self.n) * 3.1415926)
            self.ray_direction[i, 0][0] = cos_x
            self.ray_direction[i, 0][1] = sin_x

            self.ray_path[i, 0][0] = self.src_x
            self.ray_path[i, 0][1] = self.src_z

    
    @ti.kernel
    def ray_substep(self, k:ti.i32, data:ti.template()):
        for i in range(self.n):
            dx = self.ray_direction[i, k][0] * self.dt
            dz = self.ray_direction[i, k][1] * self.dt

            px = -(1.0/data[0]**3.0)*data[1] * self.dt
            pz = -(1.0/data[0]**3.0)*data[2] * self.dt

            self.ray_direction[i, k+1][0] = self.ray_direction[i, k][0] + px
            self.ray_direction[i, k+1][1] = self.ray_direction[i, k][1] + pz

            self.ray_path[i, k+1][0] = self.ray_path[i, k][0] + dx
            self.ray_path[i, k+1][1] = self.ray_path[i, k][1] + dz

            if self.ray_path[i, k+1][0] == 0.0 or self.ray_path[i, k+1][1] == 0.0:
                self.ray_path[i, k+1][0] = self.ray_path[i, k][0]
                self.ray_path[i, k+1][1] = self.ray_path[i, k][1]


    @ti.kernel
    def ray_data_tidy(self, xn:ti.i32, nx:ti.f32, nz:ti.f32):
        for i in self.ray_position:
            aa = ti.Vector([self.ray_path[xn, i][0]/nx/self.dx,1 - self.ray_path[xn, i][1]/nz/self.dz])
            self.ray_position[i] = aa
    
    def ray_paint_single(self, gui, xn, nx, nz, color, radius):
        self.ray_data_tidy(xn, nx, nz)
        print(self.ray_position)

        while True:
            gui.circles(self.ray_position.to_numpy(),color=color, radius=radius)
            gui.show()

    def ray_paint_multi(self, gui, nx, nz, color, radius):
        while True:
            for i in range(self.n-1):
                self.ray_data_tidy(i+1, nx, nz)
                gui.circles(self.ray_position.to_numpy(), color=color, radius=radius)
            gui.show()
            # print(self.ray_position)


