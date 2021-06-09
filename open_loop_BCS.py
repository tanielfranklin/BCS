#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 21:44:16 2021

@author: taniel
"""
import numpy as np
from casadi import *
from matplotlib import pyplot as plt
plt.style.use('seaborn-poster')
import timeit
import EKF
exec(compile(open('bcs_settings.py', "rb").read(), 'bcs_settings.py', 'exec')) #% Roda arquivo com modelo BCS
#%% Calculo do estacionario do modelo
print('Calculando Estacionario')

t=timeit.default_timer();
print(t)
#% Condição inicial 
fq_ss = 50; zc_ss = 50; pm_ss = 2e6; 
uss = np.array([[fq_ss],[zc_ss],[pm_ss]]); # Entradas do estacionario

xss = np.array([[8311024.82175957],[2990109.06207437],[0.00995042241351780],[50],[50]]);
yss = np.array([[6000142.88550200],[592.126490003812],[89533.9403916785],[35.8135761566714],[0.00995042241351779],[0.00995042241351780]]);
t=timeit.default_timer()-t;

#%% Configurações do NMPC
print('Configurando NMPC ')
t=timeit.default_timer();
# Definir variaveis manipuladas e controladas e disturbio externo
mv = [0,1];    #% [f, Zc]
pv = [0,1];  #% [pin, H]  #% [P, I]
#pv = [2,3];  #% [pin, H]
de = 2;      #% [pm]
tg = 2;      #% MV target
#% Parametros
ts = 1;                     #% tempo de amostragem (ts = 1 não converge)

#% Restrição
umin  = np.array([35, 0]); np.transpose(umin);  # lower bounds of inputs
umax  = np.array([65, 100]); np.transpose(umax); # upper bounds of inputs 
dumax = np.array([0.5, dzc_max]); np.transpose(dumax);  # maximum variation of input moves (ts*[0.5 dzc_max]')


#%Modelo de predição
#% Criando o objeto para predição do modelo 
# Iniciando variavel dicionário para a construção da EDO
sedo = {'x': BCS['x'], # Estados
        'p': BCS['u'], #Variáveis exogenas
        'ode': BCS['NaoLinear']['sedo_nl'] # SEDO (Gerado no bcs_settings)
        };     
#% Criando o objeto para integração da Eq_estado                                  
opt = {'tf':ts,'t0':0};                        #% opções do integrador
int_odes = integrator('int_odes','cvodes',sedo,opt); # objeto integrador 
res = int_odes(x0=BCS['x'],p=BCS['u']);             #   % solução um passo a frente
npv = len(pv); nmv = len(mv);

# Criando o objeto para solução da equação de medição
Eq_medicao = Function('Eq_medicao',[BCS['x'],BCS['u']],[BCS['y'][pv]],['x','u'],['y']);
# Criacao do objeto para simulacao do BCS Eq de estado + Eq de Medicao
Modelo_Predicao = Function('Modelo_Predicao',[BCS['x'],BCS['u']],[res['xf'],Eq_medicao(res['xf'],BCS['u'])],['xk_1','uk_1'],['xk','yk']);

# Definições do fitro de Kalman
print('Configurando EKF')
t = timeit.default_timer();

BCS['Simulation'] = Modelo_Predicao;
xmk = xss;
ymk=np.zeros((len(pv),1))
for i in pv:
    ymk[i]=yss[i]

#% Variancia da medição
V = ((0.01/3)*np.diag(ymk))**2; # +/- 1% do estado estacionario
# Variancia do modelo
W = ((0.03/3)*np.diag(xss))**2; # +/- 3% do estado estacionario
# Variancia da estimacao
Mk = W;

# Inicialização do EKF
#for i in range(0,100):
#    [xmk,ymk,Kf,Mk] = EKF.EKF(BCS,xmk,ymk,uss,W,V,Mk,ts,pv);
    
 

t = timeit.default_timer()-t;
print('Tempo:')
print(t)
#%% Simulacao
tsim = 100;
nsim=int(round(tsim/ts));       # Numero de simulacao

# Inicializa��o das variaveis
xmk = xss;           # Estados
xpk = xss;
uk_1 = uss[mv];     # MVS

# Aloca��o de variaveis
Xk = np.zeros((nx,1)); Yk = np.zeros((npv,1));
Ymk = Yk; Ys = Yk; Ymin = Yk; Ymax = Yk;
Uk = np.zeros((nmv,1));

#%h = waitbar(0,'Executando a simula��o...');
print('Simulando cenário')

for k in range(1,nsim):
    print('Tempo:',k*ts, 'k=',k)
    
    uk_1 = np.array([[60], [70]]);
    
    # Modelo n�o linear
    [xpk,ypk] = Modelo_Predicao(xpk,np.vstack((uk_1,uss[de])));
    #ypk = ypk + mvnrnd(zeros(npv,1),V)'*0;
    ypk = ypk #+ np.transpose(np.random.multivariate_normal(np.zeros((npv,1)),V))
    
    # Atualiza��o  EKF
    #[xmk,ymk,Kf,Mk] = EKF(BCS,xmk,ymk,uk,W,V,Mk,ts,pv);
    
    # Erro de predi��o
    #%ep = full(ypk) - ymk;
    
    #% Salvando as variaveis
    #%Ep(:,k) = ep;
    Xk = hcat([Xk,xpk]);
    Yk = hcat([Yk,ypk]);
    Ymk = hcat([Ymk,ymk]);
    Uk = hcat([Uk,xpk[3:5]]);
    
#%% Grafico
print('Construindo Gráficos \n')
fig1=plt.figure()
label = [r'$p_{in}(bar)$','H(m)','P','I','qc','qr' ];
xi=(np.arange(0,nsim));
for iy in range(0,npv):
    ax = fig1.add_subplot(npv,1,iy+1)

    if iy == 0: # Pin
        ax.plot(xi,(Yk[iy,:]/1e5).T, label='Medição')
        ax.plot(xi,Ymk[iy,:].T/1e5, label='EKF')
        ax.set_ylabel(label[iy])
        ax.set(xlim=(xi[0], nsim*ts))
        ax.set(ylim=(40,62))
        plt.grid(True)
    else: # H
        ax.plot(xi,Yk[iy,:].T, label='Medição')
        ax.plot(xi,Ymk[iy,:].T,label='EKF')
       # ax.set_ylabel(label[iy])
        ax.set(xlim=(xi[0], nsim*ts))
       # ax.set(ylim=(580, 850))
        plt.grid(True)
#ax.plot(xi,Yk[2,:].T, label='EKF')
ax.legend();
ax.set_xlabel('Time (nT)')
fig1.show()

#%%
Xk[0:3,-1]

#%%


fig2=plt.figure()
label = ['f(Hz)',r'$z_c$(%)'];
for iu in range(0,nmv):
    ax2=fig2.add_subplot(nmv,1,iu+1)
    ax2.plot(xi,Uk[iu,:].T, label='Medição')
    ax2.plot([1,nsim],[umin[iu], umin[iu]],'--r')
    ax2.plot([1,nsim],[umax[iu], umax[iu]],'--r', label='Restrição')
    ax2.set_ylabel(label[iu])
    ax2.set(xlim=(xi[0], nsim*ts))
    if iu==0:
        ax2.set(ylim=(30, 70))
        print(iu)
    plt.grid(True)
fig2.show()

ax2.set_xlabel('Time (nT)')
ax2.legend();

fig3,ax3=plt.subplots()
plt.grid(True)
BCS['Envelope']['fig'](ax3); # grafico do envelope
# Evolução dentro do envelope
ax3.plot(Xk[2,1:].T*3600,Yk[1,1:].T,'--k')
ax3.plot(Xk[2,1]*3600,Yk[1,1],'o')#,'MarkerFaceColor',[0,1,0],'MarkerEdgeColor',[0,0,0])
ax3.plot(Xk[2,-1]*3600,Yk[1,-1],'o')#,'MarkerFaceColor',[1,0,0],'MarkerEdgeColor',[0,0,0])
ax3.annotate('t=0',
             xy=(float(Xk[2,1]*3600),float(Yk[1,1])),
             xytext=(float(Xk[2,1]*3600)-4,float(Yk[1,1])+10),
             arrowprops=dict(facecolor='black', shrink=0.01))

ax3.annotate('t='+str(nsim),
             xy=(float(Xk[2,-1]*3600),float(Yk[1,-1])),
             xytext=(float(Xk[2,-1]*3600)-7,float(Yk[1,-1])+10),
             arrowprops=dict(facecolor='black', shrink=0.01))
plt.show()

#plotando os estados
fig4=plt.figure()
label = ['Pbh', 'Pbw', 'q', 'fq'];

for i in range(0,3):
    ax4 = fig4.add_subplot(3, 1, i+1)
    ax4.plot(xi,Xk[i,:].T,label=label[i])
    ax4.set_ylabel(label[i])
ax4.legend();
ax4.set_xlabel('Time (nT)')
fig4.show()
