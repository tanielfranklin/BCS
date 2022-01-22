import numpy as np
import math
pi=math.pi
## Constantes
g   = 9.81;   # Gravitational acceleration constant [m/s�]
Cc = 3.10049373466703e-08 ;   # Choke valve constant
#A1 = 0.008107;# Cross-section area of pipe below ESP [m�]
#A2 = 0.008107;# Cross-section area of pipe above ESP [m�]
#D1 = 0.1016;  # Pipe diameter below ESP [m]
#D2 = 0.1016;  # Pipe diameter above ESP [m]
#h1 = 200;     # Heigth from reservoir to ESP [m]
hw = 32;    # Total vertical distance in well [m]
h_r = 32;                                        # [m]      Profundidade do reservat�rio
h_p = 22.7; # [m]     # Profundidade do conjunto BCS
h_c = 0;                                         # [m]      Profundidade do choke
#Par.Vol_2.h_c = 0;                                         % [m]      Profundidade do choke
l1 =  9.3;    # Length from reservoir to ESP [m]
l2 = 22.7;    # Length from ESP to choke [m]
V1 = 4.054;   
V2 = 9.729;   
L1=l1
L2=l2
r2=0.0375;
r1=0.11;
h1=L1
V1=L1*pi*r1**2;# Pipe volume above ESP [m3]
V2=L2*pi*r2**2;# Pipe volume below ESP [m3]
f0 = 60;      # ESP characteristics reference freq [Hz]
q0_dt = 25/3600; # Downtrhust flow at f0
q0_ut = 50/3600; # Uptrhust flow at f0
Inp = 25;     # ESP motor nominal current [A]
Pnp = 13422.5982;# ESP motor nominal Power [W]
#
b1 = 1.8e9;   # Bulk modulus below ESP [Pa]
#
b2 = 1.8e9;   # Bulk modulus above ESP [Pa]
rho = 836.8898;
Anular = 0.033595; 

#M  = 1.992e8; # Fluid inertia parameters [kg/m4]
rho_1 = 836.8898;    # Density of produced fluid [kg/m�?³]
rho_2 = 836.8898;
pr = 2.1788e5;  # Reservoir pressure 
pm = 1.3394e+04;  # manifold pressure
PI = 2.7e-8; # Well productivy index [m3/s/Pa]
mu  = 0.012;  # Viscosity [Pa*s]
dfq_max = 0.5;    # m�xima varia��o em f/s
dzc_max = 1;  # m�xima varia��o em zc #/s
# tp =np.array([[1/dfq_max,1/dzc_max]]).T;  # Actuator Response time

H_aguabep = 4.330800000000000e+02; #ft
Q_aguabep = 4.401900000000000e+02; #bpd

y1 = -112.1374+6.6504*math.log(H_aguabep)+12.8429*math.log(Q_aguabep);
Q = math.exp((39.5276+26.5605*math.log(mu*1000)-y1)/51.6565); #Pa.s to Cp;
Cq = (1.0-4.0327e-3*Q-1.7240e-4*Q**2);
CH = 1.0-4.4723e-03*Q -4.1800e-05*Q**2; # 80%
#q_0a = vazao_media/Cq*(Par.f0/freq);.
# CH = -0.03*mu + 1;
# Cq = 2.7944*mu**4 - 6.8104*mu**3 + 6.0032*mu**2 - 2.6266*mu + 1;
# Cp = -4.4376*mu**4 + 11.091*mu**3 -9.9306*mu**2 + 3.9042*mu + 1;

l_bar = (l1 + l2)/2;
r_bar = (((r1*l1)+(r2*l2))/(l1+l2));
A_bar =  math.pi*r_bar**2;
rho_bar = ((rho_1*V1)+(rho_2*V2))/(V1+V2); 



A1=pi*r1**2
A2=pi*r2**2
Am=(A1+A2)/2
D1=2*r1;D2=2*r2
Lm=(l1+l2)/2
M=rho_bar*l_bar/A_bar

# =========================================================================
# Curva da bomba
# ========================================================================= 
Pot  = np.array([3.14, 3.41,3.57,3.60]).T*745.7 # Hp to Watts
Head = np.array([78255575038455.9,-243021891442.447,154711075.976357,-63654.8760768729,187.303058039876]).T
Eff =  np.array([-38000037638726.0,76885738063.1052,-104037865.052405,92277.4570774092,-0.000102311045885796]).T
#========================================
# Limites das variáveis para normalização LEA
#========================================
flim=[35,65]
zlim=[5,70]
pbhlim=(1e5,2.5e5) 
pwhlim=(2e5,11e5) 
qlim=(0.1/3600,5/3600)
#qchlim=[0.0, 0.02618969262897142]

# Calculo dos limites de normalização do HEAD e delta de press�o
def LimH(fL,qL):
    def funcaoH(fi,qi): #vazão em m3/s
        #================================
        #== Função de cálculo do H LEA
        q0 = (qi) / Cq * (f0 / fi)
        H0 =  Head[0]*q0**4 +  Head[1]*q0**3 +  Head[2]*q0**2 + Head[3]*q0 + Head[4];
        H = CH * H0 * (fi / f0) ** 2  # Head
        #================================
        return H
    fx = np.arange(fL[0],fL[1],1) # 35 a 65
    qx = np.arange(qL[0]*3600,qL[1]*3600,0.5)
    X,Y = np.meshgrid(fx, qx/3600) # grid of point
    Z = funcaoH(X, Y) # evaluation of the function on the grid
    return[min([valor for linha in Z for valor in linha]),max([valor for linha in Z for valor in linha])]

# Definimos duas variáveis para teste nas funções qch  e encontrar os limites assumindo pm fixo
def LimQch(zL,pwL):
    zL=np.array(zL)
    def funcaoqch(zi,pwi): #vazão em m3/s
        return Cc*(zi)*np.sqrt(abs(pwi - pm))

    zx = np.arange(zL[0],zL[1],5) # 35 a 65
    px = np.arange(pwL[0],pwL[1],1000)
    X,Y = np.meshgrid(zx, px) # grid of point
    W = funcaoqch(X, Y) # evaluation of the function on the grid
    return [min([valor for linha in W for valor in linha]),max([valor for linha in W for valor in linha])]

#========================================================
# Encontrar os limites dos termos não lineares de qch e H
qch_lim=LimQch(zlim,pwhlim)
H_lim=LimH(flim,qlim)
#========================================================
print(qch_lim[0]*3600,qch_lim[1]*3600,H_lim)
print(f"qch_lim={qch_lim}")
print(f"H_lim={H_lim}")
print('Dados carregados')
