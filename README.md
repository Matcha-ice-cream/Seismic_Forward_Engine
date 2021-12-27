# Seismic_Forward_Engine
Seismic Forward Engine based on taichi and python.

这是一个基于taichi与python制作的地震波场模拟的研究程序，目前来说是实现实时看到模型的结果的效果。我们希望这个程序最终可以实现交互式调参、不同方式对比等等功能，目前还在建设中。


    2021.12.16 浪了一天没写代码，明天一定写！
    
    2021.12.27 写代码没灵感QAQ


运行环境

    python 3.8

    taichi 0.8.7

在运行程序前，请确保已经拥有了以上环境。

波场与地震记录同步显示：

![demo](./image/example_20211112.gif)


生成柏林噪声(用于模型生成/射线追踪)

![demo](./image/perlin_noise.png)


生成基于munk和perlin noise的海水分布模型（静态）

![demo](./image/munk_perlin.png)

运行方式：

    python FD_wave_example.py

    python model_build_example.py
