#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 21:44:16 2021

@author: taniel
"""


from casadi import *
from matplotlib import pyplot as plt
plt.style.use('seaborn-poster')
import timeit
import EKF
exec(compile(open('bcs_settings_norm.py', "rb").read(), 'bcs_settings_norm.py', 'exec')) #% Roda arquivo com modelo BCS
#%% Calculo do estacionario do modelo
print('Calculando Estacionario')

t=timeit.default_timer();
print(t)
#% Condição inicial 
fq_ss = 50; zc_ss = 50; pm_ss = 2e6;
#% Condição inicial normalizada
fq_ss = 65; zc_ss = 100; pm_ss = 2e6;
uss = [fq_ss,zc_ss,pm_ss];
def normalizar_u(uss):
    for i in range(0,3):
        uss[i]=(uss[i]+norm_u[i,0])/norm_u[i,1]
    return uss_norm
#% Calculo do estacionario
#x0 = [0.2,0.5,0.5]

x0 = [8311024.82175957,2990109.06207437,0.00995042241351780,50,50];
#x0= [2.18143, 3.4606, 0.0241711, 50, 50]
args['lbx'][3] = uss[0];
args['ubx'][3] = uss[0];   # bounds freq. solver
args['lbx'][4]= uss[1];
args['ubx'][4] = uss[1];   # bounds zc solver
sol=solver(x0=x0, lbx=args['lbx'], ubx=args['ubx'], p=uss);
#sol=solver(x0=x0, p=uss)
sol['x']
xss=sol['x']
print(xss)


