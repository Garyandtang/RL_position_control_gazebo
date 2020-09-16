#!/usr/bin/env python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def arc_func(center_x, point):
    return (point, center_x * 1.5 / 2 * np.arctan(4 / center_x * point))

def circle_func(center_x, phase):
    radius = arc_func(center_x, center_x)[1]
    return (radius*np.cos(phase) + center_x, radius*np.sin(phase))

def eight(center_x, arc_step = 0.03, circle_step=0.2):
    step = arc_step
    cur = 0

    arc_path = []
    circle_path = []
    while cur < center_x:
        point = arc_func(center_x, cur)
        arc_path.append(point)
        cur = cur + step
        
    cur = np.pi / 2
    step = -circle_step
    while cur > -np.pi / 2:
        point = circle_func(center_x, cur)
        circle_path.append(point)
        cur = cur + step
    
    path = (arc_path + circle_path
            + list(map(lambda p: (p[0],-p[1]), arc_path[::-1]))
            + list(map(lambda p: (-p[0],p[1]), arc_path))
            + list(map(lambda p: (-p[0],p[1]), circle_path))
            + list(map(lambda p: (-p[0],-p[1]), arc_path[::-1]))
            )

    # print(list(map(lambda p: (p[0],-p[1]), arc_path[::-1])))
    return path

def display_path(path):
    x,y = zip(*path)
    plt.scatter(x,y)
    plt.axis('equal')
    plt.show()
        
if __name__ == "__main__":
    path = eight(4)
    display_path(path)