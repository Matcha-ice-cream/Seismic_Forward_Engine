import taichi as ti
from ray.RK_ray import RK_ray
from model.model_operation import getmodel
import Tools.SFE_tools as tools
import Visualization.SFE_visual as visual


ti.init(arch=ti.gpu)


src_x = 2500
src_z = 200
dt = 2
frame = 2000
n = 5
ray_cs = RK_ray(src_x, src_z, dt, frame, n)
ray_cs.RK_ray_init()



nx = 500
nz = 500
dx = 10
dz = 10
model_cs = getmodel(nx, nz, dx, dz)
model_cs.generate_rand()
perlin_nx = 50
perlin_nz = 50
model_cs.model_perlin_munk(perlin_nx, perlin_nz,400.0, 1000.0, 1500.0)

for k in range(frame):
    for i in range(n):
        model_cs.model_perlin_munk_node(perlin_nx, perlin_nz, 400.0, 1000.0, 1500.0, ray_cs.ray_path[i, k][0], ray_cs.ray_path[i, k][1])
        ray_cs.RK_ray_substep(k, model_cs.data, perlin_nx, perlin_nz, 400.0, 1000.0, 1500.0,nx, nz, dx,dz)
        # print(model_cs.data)

# print(ray_cs.ray_path[1,1][0])
gui = ti.GUI("result", (nx, nz), background_color=0x112F41)

# print(ray_cs.ray_position)

# ray_cs.ray_paint_single(gui, 2, nx,nz,0xEEEEF0, 1.5)
# ray_cs.ray_data_tidy(2, nx, nz)
# tools.export(ray_cs.ray_position, "data.txt")
ray_cs.RK_ray_paint_multi(gui,nx,nz,dx,dz,color=0xEEEEF0,radius=1)




