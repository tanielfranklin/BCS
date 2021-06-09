#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 22:29:34 2021

@author: Taniel Silva Franklin adaptado para Python do modelo de Bruno Aguiar
"""

# A ideia era cria um objeto BCS como recomenda a documenta��o do casadi.
# Assim seria f�cil fazer um import BCS, e carregar os parametros e metodos
# do BCS, o que permitiria melhores desenvolvimento, o mesmo vale para o
# NMPC. Como eu n�o conseguir fazer isso ainda, criei um .m com as
# defini�oes do BCS
## CASADI
# @Article{Andersson2018,
#   Author = {Joel A E Andersson and Joris Gillis and Greg Horn
#             and James B Rawlings and Moritz Diehl},
#   Title = {{CasADi} -- {A} software framework for nonlinear optimization
#            and optimal control},
#   Journal = {Mathematical Programming Computation},
#   Year = {In Press, 2018},
# }
#modelo
import numpy as np

# Constantes
g   = 9.81;   # Gravitational acceleration constant [m/s�]
Cc = 2e-5 ;   # Choke valve constant
A1 = 0.008107;# Cross-section area of pipe below ESP [m�]
A2 = 0.008107;# Cross-section area of pipe above ESP [m�]
D1 = 0.1016;  # Pipe diameter below ESP [m]
D2 = 0.1016;  # Pipe diameter above ESP [m]
h1 = 200;     # Heigth from reservoir to ESP [m]
hw = 1000;    # Total vertical distance in well [m]
L1 =  500;    # Length from reservoir to ESP [m]
L2 = 1200;    # Length from ESP to choke [m]
V1 = 4.054;   # Pipe volume below ESP [m3]
V2 = 9.729;   # Pipe volume above ESP [m3]
f0 = 60;      # ESP characteristics reference freq [Hz]
q0_dt = 25/3600; # Downtrhust flow at f0
q0_ut = 50/3600; # Uptrhust flow at f0
Inp = 65;     # ESP motor nominal current [A]
Pnp = 1.625e5;# ESP motor nominal Power [W]
b1 = 1.5e9;   # Bulk modulus below ESP [Pa]
b2 = 1.5e9;   # Bulk modulus above ESP [Pa]
M  = 1.992e8; # Fluid inertia parameters [kg/m4]
rho = 950;    # Density of produced fluid [kg/m�?³]
pr = 1.26e7;  # Reservoir pressure
PI = 2.32e-9; # Well productivy index [m3/s/Pa]
mu  = 0.025;  # Viscosity [Pa*s]
dfq_max = 0.5;    # m�xima varia��o em f/s
dzc_max = 1;  # m�xima varia��o em zc #/s
tp =np.array([[1/dfq_max,1/dzc_max]]).T;  # Actuator Response time 
CH = -0.03*mu + 1;
Cq = 2.7944*mu**4 - 6.8104*mu**3 + 6.0032*mu**2 - 2.6266*mu + 1;
Cp = -4.4376*mu**4 + 11.091*mu**3 -9.9306*mu**2 + 3.9042*mu + 1;
from casadi import *

# Criando simbolica
nx = 5; nu = 3; 
x = MX.sym("x",nx); # Estados
u = MX.sym("u",nu); # Exogena
dudt_max = MX.sym("dudt_max",2); # Exogena


## Montado o sistema de equa��es
# Estados
pbh = x[0]; pwh = x[1]; q = x[2]; fq = x[3]; zc = x[4];
# Entradas
fqref = u[0]; zcref = u[1]; pm = u[2];
# SEA
# Calculo do HEAD e delta de press�o
q0 = q/Cq*(f0/fq); H0 = -1.2454e6*q0**2 + 7.4959e3*q0 + 9.5970e2;
H = CH*H0*(fq/f0)**2; # Head
Pp = rho*g*H;       # Delta de press�o
# Calculo da Potencia e corrente da bomba
P0 = -2.3599e9*q0**3 -1.8082e7*q0**2 +4.3346e6*q0 + 9.4355e4;
P = Cp*P0*(fq/f0)**3; # Potencia
I = Inp*P/Pnp;      # Corrente


# Calculo da press�o de intaike
F1 = 0.158*((rho*L1*q**2)/(D1*A1**2))*(mu/(rho*D1*q))**(1/4);
F2 = 0.158*((rho*L2*q**2)/(D2*A2**2))*(mu/(rho*D2*q))**(1/4);
pin = pbh - rho*g*h1 - F1;
# Vazao do rezervatorio vazao da chocke
qr  = PI*(pr - pbh);
qc  = Cc*(zc/100)*sign((pwh - pm))*sqrt(fabs(pwh - pm));

# SEDO
dpbhdt = b1/V1*(qr - q);
dpwhdt = b2/V2*(q - qc);
dqdt = 1/M*(pbh - pwh - rho*g*hw - F1 - F2 + Pp);
dfqdt = (fqref - fq)/tp[0];
dzcdt = (zcref - zc)/tp[1];
dxdt = vertcat(dpbhdt,dpwhdt,dqdt,dfqdt,dzcdt);


# Restricao do Elemento Final
#dudt = [if_else(fabs(dfqdt)>dudt_max(1),sign(dfqdt)*dudt_max(1),dfqdt);
#        if_else(fabs(dzcdt)>dudt_max(2),sign(dzcdt)*dudt_max(2),dzcdt)];

dudt = vertcat(if_else(fabs(dfqdt)>dudt_max[0],sign(dfqdt)*dudt_max[0],dfqdt),
       if_else(fabs(dzcdt)>dudt_max[1],sign(dzcdt)*dudt_max[1],dzcdt));



# Cria��o de fun��o casadi

#Eq_Estado = Function('Eq_Estado',{x,u,dudt_max},{[dpbhdt;dpwhdt;dqdt;dudt]},{'x','u','dumax'},{'dxdt'}); % Sistema de EDO

Eq_Estado = Function('Eq_Estado', [x,u,dudt_max], [vertcat(dpbhdt,dpwhdt,dqdt,dudt)]); #entradas EDO
                      #['x','u','dumax'],['dxdt']) # Saidas EDO
# Eq_Estado = Function('Eq_Estado', [x,u,dudt_max], [[dpbhdt],[dpwhdt],[dqdt],[dudt]], #entradas EDO
#                      ['x','u','dumax'],['dxdt']) # Saidas EDO
import sys
sys.exit("Erro saindo")

