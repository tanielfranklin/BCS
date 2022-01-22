from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import deepxde as dde
from deepxde.backend import tf
exec(compile(open('param.py', "rb").read(), 'param.py', 'exec')) #% Roda arquivo com modelo BCS

def normalizar(x, xnorm):
    xc=(x-xnorm[0])/(xnorm[1]-xnorm[0])
    return xc

def Fnorm(xlim):
    # Encontrar o fator de normalização
    # tal que xb=(x-x0)/xc
    # xmin<x<xmax
    # fazendo com que 0<xb<1
    x=(xlim[0],xlim[1]-xlim[0])
    return x
def normalizar(x,xnorm):
    xs=np.zeros((nx,1))
    for i in range(0,nx):
        xs[i]=(x[i]-xnorm[i,0])/xnorm[i,1]
    return xs
def desnormalizar(x,xnorm):
    xs=np.zeros((nx,1))
    for i in range(0,nx):
        xs[i]=x[i]*xnorm[i,1]+xnorm[i,0]
    return xs
def AplicaEscala(var,i):
    aux=var*xnorm[i,1]+xnorm[i,0]
    return aux
def normaliza_u(u,unorm):
    aux=np.zeros_like(u)
    for i in range(0,len(u)):
        aux[i]=(u[i]-unorm[i,0])/unorm[i,1]
    return aux

# Valores máximos e mínimos para normalização
#Entradas
f_lim=(30,75); zclim=(0,100);pmlim=(1e6,2e6);
pbhlim=(100000,8.5e6); pwhlim=(2e6,5.2e6); qlim=(12/3600,55/3600)

nx=5

pm=2e6; #Simplificando pm fixo

unorm=np.array([Fnorm(f_lim),Fnorm(zclim)])
xnorm= np.array([Fnorm(pbhlim),Fnorm(pwhlim),Fnorm(qlim), unorm[0,:],unorm[1,:]])




# Entradas
u=[60,70,2e6] # f, zc, pm
step_u=[45,55]*unorm[0,1]+unorm[0,0]
fqref = u[0]*unorm[0,1]+unorm[0,0]; zcref = u[1]*unorm[1,1]+unorm[1,0];
pm=2e6;

#uss = np.array([[fq_ss], [zc_ss], [pm_ss]])
#u = uss
zc=u[1]

def constante(valor,time_vector):
    return np.ones_like(time_vector)*valor
# time points
maxtime = 10
time = np.linspace(0, maxtime, 200)
entrada=constante(0.4,time)# exogenous input



def ex_func(t):
    spline = sp.interpolate.Rbf(
        time, entrada, function="thin_plate", smooth=0, episilon=0
    )
    # return spline(t[:,0:])
    return spline(t)
def ex_func2(t):
    spline = sp.interpolate.Rbf(
        time, entrada, function="thin_plate", smooth=0, episilon=0
    )
    return spline(t[:, 0:])

def ED_BCS(t,x,ex):
    ## Montado o sistema de equa��es
    # Tensores (Estados)
    # pbh = x[:,0:1]
    # pwh = x[:,1:2]
    # q = x[:,2:3] #Vazão
    pbh = x[:,0:1]*xnorm[0, 1] + xnorm[0, 0]
    pwh = x[:,1:2]*xnorm[1, 1] + xnorm[1, 0]
    q = x[:,2:3]*xnorm[2, 1] + xnorm[2, 0] #Vazão
    # pbh = (x[:,0:1]- xnorm[0, 0])/xnorm[0, 1]
    # pwh = (x[:,1:2] - xnorm[1, 0])/xnorm[1, 1]
    # q = (x[:,2:3] - xnorm[2, 0])/xnorm[2, 1]  #Vazão

    #fq = x[:,3:4] # Frequencia da bomba
    #zc = x[:,4:] # Abertura da choke
    u = [0.4, 0.5, 2e6] # u normalizado    fq=ex;
    zc=u[1];
    # Calculo do HEAD e delta de press�o
    q0 = q / Cq * (f0 / fq)
    H0 = -1.2454e6 * q0 ** 2 + 7.4959e3 * q0 + 9.5970e2
    H = CH * H0 * (fq / f0) ** 2  # Head
    Pp = rho * g * H  # Delta de press�o

    # Calculo da Potencia e corrente da bomba
    P0 = -2.3599e9 * q0 ** 3 - 1.8082e7 * q0 ** 2 + 4.3346e6 * q0 + 9.4355e4
    P = Cp * P0 * (fq / f0) ** 3;  # Potencia
    I = Inp * P / Pnp  # Corrente

    # Calculo da press�o de intake
    F1 = 0.158 * ((rho * L1 * q ** 2) / (D1 * A1 ** 2)) * (mu / (rho * D1 * q)) ** (1 / 4)
    F2 = 0.158 * ((rho * L2 * q ** 2) / (D2 * A2 ** 2)) * (mu / (rho * D2 * q)) ** (1 / 4)
    pin = pbh - rho * g * h1 - F1;
    # Vazao do reservatorio vazao da choke
    qr = PI * (pr - pbh);
    qc = Cc * (zc / 100) * tf.sign((pwh - pm)) * tf.sqrt(tf.abs(pwh - pm));

    # SEDO

    dpbhdt = dde.grad.jacobian(x, t, i=0)
    dpwhdt = dde.grad.jacobian(x, t, i=1)
    dqdt = dde.grad.jacobian(x, t, i=2)
    #dfqdt = dde.grad.jacobian(x, t, i=3)
    #dzcdt = dde.grad.jacobian(x, t, i=4)

    dpbhdt = dpbhdt* xnorm[0, 1]
    dpwhdt = dpwhdt* xnorm[1, 1]
    dqdt = dqdt*xnorm[2, 1]

    dxdt = [dpbhdt-b1 / V1 * (qr - q),
            dpwhdt -b2 / V2 * (q - qc),
            dqdt - 1 / M * (pbh - pwh - rho * g * hw - F1 - F2 + Pp)
            #dfqdt - (fqref - fq) / tp[0],
            #dzcdt - (zcref - zc) / tp[1]
            ]

    return dxdt

def boundary(_, on_initial):
    return on_initial


x0 = np.float32(np.array([8311024.82175957,2990109.06207437,0.00995042241351780,50,50]))
print(normalizar(x0,xnorm))
def func(t):
    #return np.hstack((xss[0],xss[1],xss[2],xss[3],xss[4]))
    a=xss.reshape(1,3)
    return a
x0_n=np.float32(normalizar(x0,xnorm))

geom = dde.geometry.TimeDomain(0, 100)
ic1 = dde.IC(geom, lambda v: x0_n[0], boundary, component=0)
ic2 = dde.IC(geom, lambda v: x0_n[1], boundary, component=1)
ic3 = dde.IC(geom, lambda v: x0_n[2], boundary, component=2)

#
# ic1 = dde.IC(geom, lambda v: 0, boundary, component=0)
# ic2 = dde.IC(geom, lambda v: 0, boundary, component=1)
# ic3 = dde.IC(geom, lambda v: 0, boundary, component=2)
#ic4 = dde.IC(geom, lambda v: xss[3], boundary, component=3)
#ic5 = dde.IC(geom, lambda v: xss[4]+10, boundary, component=4)

data = dde.data.PDE(
    geom,
    ED_BCS,
    [ic1,ic2,ic3],
    num_domain=100,
    num_boundary=6)

# data = dde.data.PDE(
#         geom, ED_BCS, [ic1,ic2,ic3,ic4,ic5], 35, 2,  num_test=100
#     )

layer_size = [1] + [30] * 2 + [3]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)#, batch_normalization="before")
model = dde.Model(data, net)
model.compile("adam", lr=0.01)#, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=2000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
