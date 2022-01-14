import taichi as ti
from model.model_operation import getmodel


ti.init(arch=ti.gpu)

## 参数初始化
nx = 500
nz = 500
dx = 10
dz = 10


## 模型建立
model_cs = getmodel(nx, nz, dx, dz)
model_cs.model_perlin_munk(dx*5, dx*5, 1000.0, 1000.0, 1500.0)
model_v = model_cs.model_vp





