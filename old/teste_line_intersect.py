#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 23:29:48 2021

@author: taniel
"""
from shapely.geometry import Point

from shapely.geometry import LineString


# Create two circles (Polygon). `Point` gives center location and `buffer` gives size.

# See http://toblerity.org/shapely/_images/intersection-sym-difference.png

a = Point(1, 1).buffer(1.5)

b = Point(2, 1).buffer(1.5)

# Convert the Polygons into LineString's

al = LineString(list(a.exterior.coords))

bl = LineString(list(b.exterior.coords))

# Find intersection points

c = al.intersection(bl)

print(c)
