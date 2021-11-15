import taichi as py
import numpy as np

def export(arr, path):
    arr_export = arr.to_numpy()
    np.savetxt(path,arr_export)

