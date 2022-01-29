import math
import numpy as np
#from casadi import *
from casadi import sqrt as csqrt
from casadi import fabs

def EDO(x,u):
    pi=3.141592653589793    
    def Lim_c(x):
        return x[1]-x[0]
    ###############################
    # Valor normalizado dos estados
    pbh = x[0]; pwh = x[1]; q = x[2]
    ###############################
    fq = u[0]; zc = u[1]; pm=u[2]; pr=u[3]
    g   = 9.81;   # Gravitational acceleration constant [m/s�]
    Cc = 3.10049373466703e-08 ;   # Choke valve constant
    hw = 32;    # Total vertical distance in well [m]
    h_r = 32;                                        # [m]      Profundidade do reservat�rio
    h_p = 22.7; # [m]     # Profundidade do conjunto BCS
    h_c = 0;                                         # [m]      Profundidade do choke
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
    b1 = 1.8e9;   # Bulk modulus below ESP [Pa]
    b2 = 1.8e9;   # Bulk modulus above ESP [Pa]
    rho = 836.8898;
    Anular = 0.033595; 
    rho_1 = 836.8898;    # Density of produced fluid [kg/m�?³]
    rho_2 = 836.8898;
    #pr = 2.1788e5;  # Reservoir pressure 
    #pm = 1.3394e+04;  # manifold pressure
    PI = 2.7e-8; # Well production index [m3/s/Pa]
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
    # Entradas
    # Valores máximos e mínimos para normalização
    #Entradas - conforme binder e pavlov
    #========================================
    pbc=Lim_c(pbhlim)
    pwc=Lim_c(pwhlim)
    qc=Lim_c(qlim)
    pbmin=pbhlim[0]
    pwmin=pwhlim[0]
    qmin=qlim[0]
  
    #=============================================
    # Computing HEAD and pump pressure gain of LEA
    q0 = (qc*q+qmin) / Cq * (f0 / fq)
    H0 =  Head[0]*q0**4 +  Head[1]*q0**3 +  Head[2]*q0**2 + Head[3]*q0 + Head[4];
    H = CH * H0 * (fq / f0) ** 2  # Head
    Pp = rho * g * H  # Dp
    #==============================================
    # Electrical power and electrical current computing
    P0 = -2.3599e9 * q0 ** 3 - 1.8082e7 * q0 ** 2 + 4.3346e6 * q0 + 9.4355e4
    #P = Cp * P0 * (fq / f0) ** 3;  
    #I = Inp * P / Pnp  
    # Computing two volumes frictions in LEA piping
    qan=q*qc+qmin # non normalized flow
    Re =(4*rho*qan)/(0.219*pi*mu); # Assuming volumes density are identicals
    fric=64/Re 
    F1 = (fric*qan**2*rho)/(2*pi*r1**3) #Frictional pressure drop above ESP (Assuming laminar flow)
    F2 = (fric*qan**2*rho)/(2*pi*r2**3) #Frictional pressure drop below ESP (Assuming laminar flow)
    # Computing Reservoir flow
    qr = PI * (pr - (pbh*pbc+pbmin));
    # Computing flow across Choke valvule
    qch = Cc * (zc) * csqrt(fabs(pwh*pwc+pwmin - pm));
    #============================================
    F1c=2.92634e-05
    F2c=0.000738599
    Hc=215.9226497597185
    qcc=0.0020328441729756536
    F1lim=(0.000439365,0.000439365)
    F2lim=(0.0110894,0.0110894)
    qch_lim=(6.69674349099543e-5, 0.0020998116078856078)
    H_lim=(-11.492505101438962, 204.43014465827954)
    #Normalizing nonlinear terms
    ##########################
    qch=(qch-qch_lim[0])/qcc
    F1=(F1-F1lim[0])/F1c
    F2=(F2-F2lim[0])/F2c
    H=(H-H_lim[0])/Hc
    ###########################
    # Computing intake pressure
    pin = pbh*pbc+pbmin - rho * g * h1 - F1*F1c+F1lim[0];

    #xss=np.float32(np.array([2.0197e5,4.9338e5,4.2961e-4]));
    # xss=x_0;uss=u_0
    dpbhdt = (1/pbc)*b1/V1*(qr - (q*qc+qmin))
    dpwhdt = (1/pwc)*b2/V2*((q*qc+qmin) - (qcc*qch+qch_lim[0]))
    dqdt = (1/(qc*M))*(pbh*pbc+pbmin + rho * g * (H*Hc+H_lim[0]) - (pwh*pwc+pwmin) - rho*g*hw - (F1c*F1+F1lim[0]) - (F2c*F2+F2lim[0]))
    return dpbhdt,dpwhdt,dqdt,pin,H*Hc+H_lim[0]