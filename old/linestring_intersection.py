#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 23:50:50 2021

@author: taniel
"""

from matplotlib import pyplot
import numpy as np
from shapely.geometry import LineString,Point
from shapely.figures import SIZE, set_limits, plot_coords, plot_bounds, plot_line_issimple

COLOR = {
    True:  '#6699cc',
    False: '#ffcc33'
    }

def v_color(ob):
    return COLOR[ob.is_simple]

def plot_coords(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, 'o', color='#999999', zorder=1)

def plot_bounds(ax, ob):
    x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, 'o', color='#000000', zorder=1)

def plot_line(ax, ob):
    x, y = ob.xy
    ax.plot(x, y, color=v_color(ob), alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

fig = pyplot.figure(1, figsize=SIZE, dpi=90)

# 1: simple line
ax = fig.add_subplot(131)
line1 = LineString([(-1, 0), (-0.5, 1), (0, 2), (2, 2), (3, 1), (3.5, 0)])
#line2 = LineString([(0, 1), (0, 1.5), (0.5, 1), (2, 0.5), (3, 2.5), (3.5, 2.4)])
p=[Point(1,2),Point(2,3)];
line2 = LineString(p);


plot_coords(ax, line1)
plot_bounds(ax, line1)
plot_line_issimple(ax, line1, alpha=0.7)

ax.set_title('a) simple')

set_limits(ax, -1, 4, -1, 3)

#2: complex line
ax = fig.add_subplot(132)
#line2 = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (-1, 1), (1, 0)])
c=line2.intersection(line1)
plot_coords(ax, line2)
plot_bounds(ax, line2)
plot_line_issimple(ax, line2, alpha=0.7)

ax.set_title('b) complex')

set_limits(ax, -1, 4, -1, 3)

#3: complex line
ax = fig.add_subplot(133)
c=line2.intersection(line1)
# plot_coords(ax, c)
# plot_bounds(ax, c)
if c.geom_type == 'MultiPoint':
    print("É lista")
    for ob in c:
        x, y = ob.xy
        if len(x) == 1:
            ax.plot(x, y, 'o', zorder=2)
        else:
            ax.plot(x, y, alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
else:
    print("Não é lista")
    x, y = c.xy
    ax.plot(x, y, 'o', zorder=2)
ax.set_title('c) intersection')

set_limits(ax, -1, 4, -1, 3)

pyplot.show()