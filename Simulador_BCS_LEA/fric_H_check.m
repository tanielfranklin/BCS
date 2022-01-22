% =========================================================================
% Define a fric��o, Deltap da bomba e corre��o da viscosidade
% =========================================================================
Par =  dados_LEA_Par;                  % Par�metros do BCS LEA                               
Cexp = dados_LEA_Exp;                  % Dados experimentais do BCS LEA   
nsim = Cexp.tempo;                     % [s] Tempo de simula��o 
Ts =   Cexp.Ts;                        % [s] Per�odo de amostragem

xss=[1.67000008e+05, 9.91387081e+05, 8.57890646e-04];
uss=[6.0000000e+01, 2.6007870e+01, 1.3936520e+04, 1.9081252e+05];
Pmanifold=uss(3);
pressao_choke=xss(2);
valv=uss(2);

vazao_media=xss(3);
visc=0.012;
freq=uss(1);
    Re_1 = (4*Par.Vol_1.rho_1*vazao_media)/(0.219*pi*visc);
    Re_2 = (4*Par.Vol_2.rho_2*vazao_media)/(0.219*pi*visc);
if Re_1<4000
    fric_1 = 64/Re_1;
    fric_2 = 64/Re_2;
else
    fric_1 = 0.36*Re_1^(-0.25);
    fric_2 = 0.36*Re_2^(-0.25);
end
    F_1 = (fric_1*vazao_media^2*Par.Vol_1.rho_1)/(2*pi*Par.Vol_1.r1^3)
    F_2 = (fric_2*vazao_media^2*Par.Vol_2.rho_2)/(2*pi*Par.Vol_2.r2^3)
    
H_aguabep = 4.330800000000000e+02; %ft
Q_aguabep = 4.401900000000000e+02; %bpd
y = -112.1374+6.6504*log(H_aguabep)+12.8429*log(Q_aguabep);
Q = exp((39.5276+26.5605*log(visc*1000)-y)/51.6565); %Pa.s to Cp;
Cq = (1.0-4.0327e-03*Q-1.7240e-04*Q^2);
CH = 1.0-4.4723e-03*Q -4.1800e-05*Q^2; % 80%
q_0a = vazao_media/Cq*(Par.f0/freq);
H_0 =  Par.Head(1)*q_0a^4 +  Par.Head(2)*q_0a^3 +  Par.Head(3)*q_0a^2 + Par.Head(4)*q_0a + Par.Head(5);
H = (CH*H_0*(freq/Par.f0)^2);
dp = H*Par.Vol_1.rho_1*Par.g   

rho=Par.Vol_1.rho_1
L1=Par.Vol_1.l1;L2=Par.Vol_2.l2;
l1 =  9.3;    %% Length from reservoir to ESP [m]
l2 = 22.7;    %% Length from ESP to choke [m]
V1 = 4.054;   
V2 = 9.729;   
L1=l1
L2=l2
r2=0.0375;
r1=0.11;
h1=L1
V1=L1*pi*r1^2;% Pipe volume above ESP [m3]
V2=L2*pi*r2^2;% Pipe volume below ESP [m3]
f0 = 60;      % ESP characteristics reference freq [Hz]
q0_dt = 25/3600; % Downtrhust flow at f0
q0_ut = 50/3600; % Uptrhust flow at f0
Inp = 25;     % ESP motor nominal current [A]
Pnp = 13422.5982;% ESP motor nominal Power [W]
%
b1 = 1.8e9;   % Bulk modulus below ESP [Pa]
%
b2 = 1.8e9;   % Bulk modulus above ESP [Pa]
rho = 836.8898;
Anular = 0.033595; 



 
%M  = 1.992e8; % Fluid inertia parameters [kg/m4]
rho_1 = 836.8898;    % Density of produced fluid [kg/m�?³]
rho_2 = 836.8898;
pr = 2.1788e5;  % Reservoir pressure 
pm = 1.3394e+04;  % manifold pressure
PI = 2.7e-8; % Well productivy index [m3/s/Pa]
mu  = 0.012;  % Viscosity [Pa*s]
dfq_max = 0.5;    % m�xima varia��o em f/s
dzc_max = 1;  % m�xima varia��o em zc %/s
% tp =np.array([[1/dfq_max,1/dzc_max]]).T;  % Actuator Response time


%q_0a = vazao_media/Cq*(Par.f0/freq);.
% CH = -0.03*mu + 1;
% Cq = 2.7944*mu^4 - 6.8104*mu^3 + 6.0032*mu^2 - 2.6266*mu + 1;
% Cp = -4.4376*mu^4 + 11.091*mu^3 -9.9306*mu^2 + 3.9042*mu + 1;

l_bar = (l1 + l2)/2;
r_bar = (((r1*l1)+(r2*l2))/(l1+l2));
A_bar =  pi*r_bar^2;
rho_bar = ((rho_1*V1)+(rho_2*V2))/(V1+V2); 



A1=pi*r1^2
A2=pi*r2^2
Am=(A1+A2)/2
D1=2*r1;D2=2*r2
Lm=(l1+l2)/2
M=rho_bar*l_bar/A_bar
vazao_choke = ((valv).*((pressao_choke-Pmanifold).^0.5))*Par.k_choke
q=vazao_media
F1 = 0.158 * ((rho * L1 * ((q))^2.0) / (D1 * A1^2.0)) * (mu / (rho * D1 * ((q))))^(1.0/4.0);
F2 = 0.158 * ((rho * L2 * ((q))^2.0) / (D2 * A2^2.0)) * (mu / (rho * D2 * ((q))))^(1.0/4.0);

F1,F_1