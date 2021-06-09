from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import deepxde as dde
from deepxde.backend import tf
exec(compile(open('param.py', "rb").read(), 'param.py', 'exec')) #% Roda arquivo com modelo BCS

# time points
maxtime = 10
time = np.linspace(0, maxtime, 200)

def ex_func(t):
    spline = sp.interpolate.Rbf(
        time, entrada, function="thin_plate", smooth=0, episilon=0
    )
    # return spline(t[:,0:])
    return spline(t)

def ED_BCS(t,x):
    ## Montado o sistema de equa��es
    # Tensores (Estados)
    pbh = x[:,0:1]
    pwh = x[:,1:2]
    q = x[:,2:3] #Vazão
    fq = x[:,3:4] # Frequencia da bomba
    zc = x[:,4:] # Abertura da choke

    # # Entradas
    # resposta livre entradas nulas
    u=np.array([0,0,0])
    # % Condição inicial
    fq_ss = 50;
    zc_ss = 50;
    pm_ss = 2e6;
    uss = np.array([[fq_ss], [zc_ss], [pm_ss]]);
    u=uss;
    fqref = u[0]
    zcref = u[1]
    pm = u[2]


    # Calculo do HEAD e delta de press�o
    q0 = q / Cq * (f0 / fq);
    H0 = -1.2454e6 * q0 ** 2 + 7.4959e3 * q0 + 9.5970e2;
    H = CH * H0 * (fq / f0) ** 2;  # Head
    Pp = rho * g * H;  # Delta de press�o

    # Calculo da Potencia e corrente da bomba
    P0 = -2.3599e9 * q0 ** 3 - 1.8082e7 * q0 ** 2 + 4.3346e6 * q0 + 9.4355e4;
    P = Cp * P0 * (fq / f0) ** 3;  # Potencia
    I = Inp * P / Pnp;  # Corrente

    # Calculo da press�o de intake
    F1 = 0.158 * ((rho * L1 * q ** 2) / (D1 * A1 ** 2)) * (mu / (rho * D1 * q)) ** (1 / 4);
    F2 = 0.158 * ((rho * L2 * q ** 2) / (D2 * A2 ** 2)) * (mu / (rho * D2 * q)) ** (1 / 4);
    pin = pbh - rho * g * h1 - F1;
    # Vazao do reservatorio vazao da choke
    qr = PI * (pr - pbh);
    qc = Cc * (zc / 100) * tf.sign((pwh - pm)) * tf.sqrt(tf.abs(pwh - pm));

    # SEDO

    dpbhdt = dde.grad.jacobian(x, t, i=0)
    dpwhdt = dde.grad.jacobian(x, t, i=1)
    dqdt = dde.grad.jacobian(x, t, i=2)
    dfqdt = dde.grad.jacobian(x, t, i=3)
    dzcdt = dde.grad.jacobian(x, t, i=4)

    dxdt = [dpbhdt-b1 / V1 * (qr - q),
            dpwhdt -b2 / V2 * (q - qc),
            dqdt - 1 / M * (pbh - pwh - rho * g * hw - F1 - F2 + Pp),
            dfqdt - (fqref - fq) / tp[0],
            dzcdt - (zcref - zc) / tp[1]
            ]

    return dxdt

def boundary(_, on_initial):
    return on_initial

xss = np.array([[8311024.82175957],[2990109.06207437],[0.00995042241351780],[50],[50]]);
yss = np.array([[6000142.88550200],[592.126490003812],[89533.9403916785],[35.8135761566714],[0.00995042241351779],[0.00995042241351780]]);

xss = np.float32(np.array([8311024.82175957,2990109.06207437,0.00995042241351780,50,50]))

def func(t):
    #return np.hstack((xss[0],xss[1],xss[2],xss[3],xss[4]))
    a=xss.reshape(1,5)
    return a

geom = dde.geometry.TimeDomain(0, 3)
ic1 = dde.IC(geom, lambda v: xss[0], boundary, component=0)
ic2 = dde.IC(geom, lambda v: xss[1], boundary, component=1)
ic3 = dde.IC(geom, lambda v: xss[2], boundary, component=2)
ic4 = dde.IC(geom, lambda v: xss[3], boundary, component=3)
ic5 = dde.IC(geom, lambda v: xss[4]+10, boundary, component=4)

data = dde.data.PDE(
        geom, ED_BCS, [ic1,ic2,ic3,ic4,ic5], 100, 2,
    #solution=func,
    num_test=500
    )

# data = dde.data.PDE(
#         geom, ED_BCS, [ic1,ic2,ic3,ic4,ic5], 35, 2,  num_test=100
#     )

layer_size = [1] + [50] * 3 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)
model.compile("adam", lr=0.001)#, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=2000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
