#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 22:29:34 2021

@author: Taniel Silva Franklin adaptado para Python do modelo em matlab desenvolvido
por Bruno Aguiar
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
fq=fqref
zc=zcref

#tamanho característico das variáveis
qr_c=50;q_c=50; zc_c=100; pbh_c=1e6; pwh_c=1e6; fq_c=50
qc_c=1; P_c=50000;I_c=50;pin_c=50;H_c=700;
#

# q_b=q/q_c
# pbh_b=pbh/pbh_c
# pwh_b=pwh/pwh_c
# zc_b=zc/zc_c
#
# q=q*q_c
# pbh=pbh*pbh_c
# pwh=pwh*pwh_c
# zc=zc*zc_c







# ###########  fator de mudança de escala
# fq_b=fq/fq_c;fq=fq*fq_c;

# SEA
# Calculo do HEAD e delta de press�o
q0 = q/Cq*(f0/fq); H0 = -1.2454e6*q0**2 + 7.4959e3*q0 + 9.5970e2;
H = CH*H0*(fq/f0)**2; # Head
# ###########  fator de mudança de escala
# H_b=H/H_c;H=H*H_c;
###########
Pp = rho*g*H;       # Delta de press�o
# Calculo da Potencia e corrente da bomba
P0 = -2.3599e9*q0**3 -1.8082e7*q0**2 +4.3346e6*q0 + 9.4355e4;
P = Cp*P0*(fq/f0)**3; # Potencia
# ###########  fator de mudança de escala
# P_b=P/P_c;P=P*P_c;
#
I = Inp*P/Pnp;      # Corrente
# ###########  fator de mudança de escala
# I_b=I/I_c;I=I*I_c

# Calculo da press�o de intake
F1 = 0.158*((rho*L1*q**2)/(D1*A1**2))*(mu/(rho*D1*q))**(1/4);
F2 = 0.158*((rho*L2*q**2)/(D2*A2**2))*(mu/(rho*D2*q))**(1/4);
pin = pbh - rho*g*h1 - F1;
#pin_b=pin/pin_c;pin=pin*pin_c

# Vazao do reservatorio vazao da choke
qr  = PI*(pr - pbh);
# qr_b=qr/qr_c; qr=qr*qr_c;

qc  = Cc*(zc/100)*sign((pwh - pm))*sqrt(fabs(pwh - pm));
# qc_b=qc/qc_c;qc=qc*qc_c
# xnorm=[pin_b,H_b,P_b,I_b,qc_b,qr_b]
# SEDO
dpbhdt = b1/V1*(qr - q);
dpwhdt = b2/V2*(q - qc);
dqdt = 1/M*(pbh - pwh - rho*g*hw - F1 - F2 + Pp);
dfqdt = (fqref - fq)/tp[0];
dzcdt = (zcref - zc)/tp[1];
dxdt = vertcat(dpbhdt,dpwhdt,dqdt,dfqdt,dzcdt);


# Restricao do Elemento Final
dudt = vertcat(if_else(fabs(dfqdt)>dudt_max[0],sign(dfqdt)*dudt_max[0],dfqdt),
       if_else(fabs(dzcdt)>dudt_max[1],sign(dzcdt)*dudt_max[1],dzcdt));

#Normaliza SEDO

def normalizar(x, xnorm):
    xc=(x-xnorm[0])/(xnorm[1]-xnorm[0])
    return xc

#pin*pin_b,H*H_b,P*P_b,I*I_b,qc*qc_b,qr*qr_b

#dxdt = dxdt.T@xnorm; # multiplicar  vetor de estados por fator de normalização



# Cria��o de fun��o casadi

Eq_Estado = Function('Eq_Estado',[x,u,dudt_max],[vertcat(dxdt[0:3],dudt)],['x','u','dumax'],['dxdt']); # Sistema de EDO

#Eq_Estado = Function('Eq_Estado', [x,u], [vertcat(dpbhdt,dpwhdt,dqdt)]); #entradas EDO
                      #['x','u','dumax'],['dxdt']) # Saidas EDO
# Eq_Estado = Function('Eq_Estado', [x,u,dudt_max], [[dpbhdt],[dpwhdt],[dqdt],[dudt]], #entradas EDO
#                      ['x','u','dumax'],['dxdt']) # Saidas EDO



# % Equa��o de medi��o
y=vertcat(pin,H,P,I,qc,qr);
ny = y.size1()
sea_nl = Function('sea_nl',[x,u],[y,pin,H,P,I,qc,qr],\
                  ['x','u'],['y','pin','H','P','I','qc','qr']); # Sistema de Eq. Algebricas variaveis de sa�da


# % Modelo n�o linear
# BCS.x = x; BCS.nx = nx;
# BCS.u = u; BCS.nu = nu;
# BCS.y = y; BCS.ny = ny;
# BCS.NaoLinear.sedo_nl = Eq_Estado(x,u,np.transpose([dfq_max,dzc_max]));
# BCS.NaoLinear.sea_nl = sea_nl;

BCS={
     'x': x,
     'u': u,
     'y': y,
     'nx': nx,
     'nu': nu,
     'ny': ny,
     'NaoLinear': {'sedo_nl': Eq_Estado(x,u,[dfq_max,dzc_max]),
                   'sea_nl': sea_nl}
}



H0_dt = -1.2454e6*q0_dt**2 + 7.4959e3*q0_dt + 9.5970e2;
H0_dt = CH*H0_dt*(f0/f0)**2;
H0_ut = -1.2454e6*q0_ut**2 + 7.4959e3*q0_ut + 9.5970e2;
H0_ut = CH*H0_ut*(f0/f0)**2;


f = np.linspace(30,70,1000); #% Hz
H_ut = H0_ut*(f/f0)**2;
H_dt = H0_dt*(f/f0)**2;

Qdt = q0_dt*f/f0;
Qut = q0_ut*f/f0;



flim = np.arange(35,70,5);
qop = np.linspace(0,q0_ut*flim[-1]/f0,1000); # m3/s
#qop=np.transpose(qop);


Hop = np.zeros((len(flim),len(qop)));


for i in range(0,len(flim)):
    q0 = qop/Cq*(f0/flim[i]);
    H0 = -1.2454e6*q0**2 + 7.4959e3*q0 + 9.5970e2;
    Hop[i,:] = CH*H0*(flim[i]/f0)**2;
    #print(i)
    


from shapely.geometry import LineString,Point,MultiPoint

# Calculo dos pontos de interse��o para delimita��o da regi�o
#points1=np.zeros((1000,1));
points1=[];points2=[];points4=[];points6=[];#points8=[];

for ind in range(0,999):
    points1.append((qop[ind]*3600,Hop[0,ind]));
    points2.append((Qdt[ind]*3600,H_dt[ind]));#2 e 3 são iguais
    #points3.append((Qdt[ind]*3600,H_dt[ind]));#2 e 3 são iguais
    points4.append((qop[ind]*3600,Hop[-1,ind])); # igual a 5
    #points5.append((qop[ind]*3600,Hop[-1,ind]));
    points6.append((Qut[ind]*3600,H_ut[ind]));
    #points7.append((Qut[ind]*3600,H_ut[ind])); #Igual ao 6
    #points8.append((qop[ind]*3600,Hop[1,ind])); #Igual ao 1
    
line1=LineString(points1); 
line2=LineString(points2); #igual a line3
line4=LineString(points4);
line6=LineString(points6);


ip=np.zeros((4,2));    
[ip[0,0],ip[0,1]] = [line2.intersection(line1).x, line2.intersection(line1).y];
[ip[1,0],ip[1,1]] = [line4.intersection(line2).x, line4.intersection(line2).y];
[ip[2,0],ip[2,1]] = [line6.intersection(line4).x, line6.intersection(line4).y];
[ip[3,0],ip[3,1]] = [line6.intersection(line1).x, line6.intersection(line1).y];
                  

# Ajuste do polinomio de frequencia maxima 65 Hz
p_35hz = np.polyfit(qop*3600,Hop[0,:],3);
H_35hz = lambda qk: p_35hz@np.vstack((cumprod_reverse(np.tile(qk,(len(p_35hz)-1,1)),0),np.ones((1,len(qk)))));
q_35hz = np.linspace(ip[0,0],ip[3,0],100);
# Ajuste do polinomio de frequencia minima 35 Hz
p_65hz = np.polyfit(qop*3600,Hop[-1,:],3);
H_65hz = lambda qk: p_65hz@np.vstack((cumprod_reverse(np.tile(qk,(len(p_65hz)-1,1)),0),np.ones((1,len(qk)))));
q_65hz = np.linspace(ip[1,0],ip[2,0],100);
# Ajuste do polinomio de Downtrhust
p_dt = np.polyfit(Qdt*3600,H_dt,2);
H_dt = lambda qk: p_dt@np.vstack((cumprod_reverse(np.tile(qk,(len(p_dt)-1,1)),0),np.ones((1,len(qk)))));
q_dt = np.linspace(ip[0,0],ip[1,0],100);
# Ajuste do polinomio de Uptrhust
p_ut = np.polyfit(Qut*3600,H_ut,2);
H_ut = lambda qk: p_ut@np.vstack((cumprod_reverse(np.tile(qk,(len(p_ut)-1,1)),0),np.ones((1,len(qk)))));
q_ut = np.linspace(ip[3,0],ip[2,0],100);

# % Constu��o da figura
# BCS.Envelope.fig = @(aux) plot(q_35hz,H_35hz(q_35hz),':r',q_65hz,H_65hz(q_65hz),':r',q_ut,H_ut(q_ut),':r',q_dt,H_dt(q_dt),':r','LineWidth',2);
# BCS.Envelope.ip = ip;
# BCS.Envelope.fBounds = struct('H_35hz',H_35hz,'H_65hz',H_65hz,'H_dt',H_dt,'H_ut',H_ut);
# % Funa��o para a avalia��o dos limites dada uma vaz�o.
# BCS.Envelope.Hlim = @(qk) BoundHead(qk*3600,ip,BCS.Envelope.fBounds);


def grafico_envelope(ax):
    ax.plot(q_35hz,H_35hz(q_35hz),':r'); 
    ax.plot(q_65hz,H_65hz(q_65hz),':r');
    ax.plot(q_ut,H_ut(q_ut),':r');
    ax.plot(q_dt,H_dt(q_dt),':r');
    ax.set_xlabel(r'$q_p (m^3/h)$')
    ax.set_ylabel('H (m)')
    #ax.set(ylim=(ip, 70))

#% Constu��o da figura
#figBCS=lambda aux: pyplot(q_35hz,H_35hz(q_35hz),':r',q_65hz,H_65hz(q_65hz),':r',q_ut,H_ut(q_ut),':r',q_dt,H_dt(q_dt),':r','LineWidth',2)
fBounds = {'H_35hz': H_35hz,
           'H_65hz': H_65hz,
           'H_dt': H_dt,
           'H_ut': H_ut}
BCS['Envelope'] = {'fig': grafico_envelope, #lambda aux: plt.plot(q_35hz,H_35hz(q_35hz),':r',q_65hz,H_65hz(q_65hz),':r',q_ut,H_ut(q_ut),':r',q_dt,H_dt(q_dt),':r','LineWidth',2),
                   'ip': ip,
                   'fbounds': fBounds}

    
    

# % Funa��o para a avalia��o dos limites dada uma vaz�o.
BCS['Envelope']['Hlim']= lambda qk: BoundHead(qk*3600,ip,BCS['Envelope']['fBounds']);
#


#%% Subrotina
def BoundHead(qk,ip,bounds):
    if qk < ip[0,0]:
        Hlim = [ip[0,1],ip[0,1]];
    elif qk < ip[1,0]:
        Hlim = [[bounds['H_35hz'][qk]],[bounds['H_dt'][qk]]];
    elif qk < ip[3,0]:
        Hlim = [[bounds['H_35hz'][qk]],[bounds['H_65hz'][qk]]];
    elif qk < ip[2,0]:
        Hlim = [[bounds['H_ut'][qk]],[[bounds['H_65hz'][qk]]]];
    else:
        Hlim = [ip[2,1],ip[2,1]];
    
    return Hlim

def cumprod_reverse (A,n):
    #Como não havia cumprod reverse no python tivemos que contornar
    if n==0:
        return np.flipud(np.cumprod(np.flipud(A),n));
    elif n==1:
        return np.fliplr(np.cumprod(np.fliplr(A),n));
    else:
        print('Erro em n - cumprod reverse')
    
    

# %% Calulo do estacionario
# % Func��o objetivo
# dxdt_0 = Eq_Estado(BCS.x,BCS.u,[dfq_max;dzc_max]);
# J = sum(dxdt_0.^2);

#%% Calculo do estacionario
#% Func��o objetivo
dxdt_0 = Eq_Estado(BCS['x'], BCS['u'],[dfq_max,dzc_max]);
J = sum1(dxdt_0**2);

#% Otimizador
opt={
     'ipopt':{
         'print_level':0,
         'acceptable_tol':1e-8,
         'acceptable_obj_change_tol':1e-6,
         'max_iter':50
         },
     'print_time':0,
     }

opt['ipopt']['print_level']=0;# %0,3
opt['print_time']=0;
opt['ipopt']['acceptable_tol']=1e-8;
opt['ipopt']['acceptable_obj_change_tol']=1e-6;
opt['ipopt']['max_iter']=50;

MMQ = {'f':J,
       'x':BCS['x'],
       'p':BCS['u']
       }
nlp={'x':vertcat(BCS['x'],BCS['u']), 'f':J}
solver = nlpsol('solver', 'ipopt', MMQ, opt);
#solver = nlpsol('solver', 'ipopt',nlp)

# % Resti��o das variaveis de decis�o
# % minimo
# args.lbx = zeros(nx,1);
# % m�ximo
# args.ubx = inf(nx,1);

# Restrições das variaveis de decis�o
# minimo
args={
      'lbx': np.zeros((nx,1)),
# m�ximo
      'ubx':np.full((nx, 1), np.inf)
      }


# Solu��o do otimizador
sol=solver(x0=BCS['x'], lbx=args['lbx'], ubx=args['ubx'], p=BCS['u']);
yss=sea_nl(sol['x'],BCS['u'])
    
Estacionario = Function('Estacionario',[BCS['x'],BCS['u']],\
    [sol['x'],yss[0]],\
    ['x0','uss'],['xss','yss']);
    
BCS['Estacionario'] = Estacionario;

# %% Lineariza��o
# % Jacobiano em rela��o ao estado e entradas
# A = jacobian(dxdt,x); B = jacobian(dxdt,u); C = jacobian(y,x); D = jacobian(y,u);

#%% Lineariza��o
#% Jacobiano em rela��o ao estado e entradas
A = jacobian(dxdt,x); B = jacobian(dxdt,u); C = jacobian(y,x); D = jacobian(y,u);


# % Funcao para gerar as matrizes da linearizacao com relacao aos estados e entrada no estacionario
# % Em unidade de engenharia (se necessario)
# Linearizacao = Function('Linearizacao',{x,u},{A,B,C,D},{'x','u'},{'A','B','C','D'});
# BCS.Linearizacao = Linearizacao;

#% Funcao para gerar as matrizes da linearizacao com relacao aos estados e entrada no estacionario
#% Em unidade de engenharia (se necessario)
Linearizacao = Function('Linearizacao',[x,u],[A,B,C,D],['x','u'],['A','B','C','D']);
BCS['Linearizacao'] = Linearizacao;

#% Normaliza��o do modelo linearizado
An = MX.zeros(A.shape); Bn = MX.zeros(B.shape);

for ix in range(0,nx):
    An[ix,:] = x.T/x[ix]
    Bn[ix,:] = u.T/x[ix]

An = A*An; Bn = B*Bn
Cn = MX.zeros(C.shape); Dn = MX.zeros(D.shape)
for iy in range(0,ny):
    Cn[iy,:] = x.T/y[iy]
    Dn[iy,:] = u.T/y[iy]

Cn = C*Cn; Dn = D*Dn;

# % Fun��o para gerar matrizes normalizadas em rela��o ao estacion�rio
# Normalizacao = Function('Normalizacao',{x,u},{An,Bn,Cn,Dn},{'x','u'},{'An','Bn','Cn','Dn'});
# BCS.Normalizacao = Normalizacao;

#% Fun��o para gerar matrizes normalizadas em rela��o ao estacion�rio
Normalizacao = Function('Normalizacao',[x,u],[An,Bn,Cn,Dn],['x','u'],['An','Bn','Cn','Dn']);
BCS['Normalizacao'] = Normalizacao;

print('BCS settings loaded')