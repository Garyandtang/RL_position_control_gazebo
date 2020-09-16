#!/usr/bin/env python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def arc_func(center_x, point):
    return (point, center_x * 1.5 / 2 * np.arctan(4 / center_x * point))

def circle_func(center_x, phase):
    radius = arc_func(center_x, center_x)[1]
    return (radius*np.cos(phase) + center_x, radius*np.sin(phase))

def ellipse_func(a_axis_len, b_axis_len, center_x, phase):
    return (a_axis_len * np.cos(phase) + center_x, b_axis_len*np.sin(phase))

def eight_ellipse(center_x, a_axis_len, b_axis_len, arc_step = 0.03, circle_step=0.2, ellipse_step = 0.2):
    step = arc_step
    cur = 0

    arc_path = []
    circle_path = []
    ellipse_path = []
    while cur < center_x:
        point = arc_func(center_x, cur)
        arc_path.append(point)
        cur = cur + step
        
    cur = np.pi / 2
    step = -circle_step
    while cur > 0:
        point = circle_func(center_x, cur)
        circle_path.append(point)
        cur = cur + step

    cur = 0
    step = -ellipse_step
    while cur > -2 * np.pi:
        point = ellipse_func(a_axis_len, b_axis_len, 2*center_x - a_axis_len, cur)
        ellipse_path.append(point)
        cur = cur + step
    
    path = (arc_path + circle_path + ellipse_path
            + list(map(lambda p: (p[0],-p[1]), circle_path[::-1]))
            + list(map(lambda p: (p[0],-p[1]), arc_path[::-1]))
            + list(map(lambda p: (-p[0],p[1]), arc_path))
            + list(map(lambda p: (-p[0],p[1]), circle_path))
            + list(map(lambda p: (-p[0],-p[1]), circle_path[::-1]))
            + list(map(lambda p: (-p[0],-p[1]), arc_path[::-1]))
            )
    return path

def display_path(path):
    x,y = zip(*path)
    plt.scatter(x,y)
    plt.axis('equal')
    plt.show()
        
if __name__ == "__main__":
    path = eight_ellipse(0.4, 1.2, 0.8, arc_step = 0.03, circle_step=0.2, ellipse_step = 0.2)
    display_path(path)