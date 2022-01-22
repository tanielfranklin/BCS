function [dydt]=dinamica_BCS(t,xmk,duk,Par)
% =========================================================================
% Define as entradas 
% =========================================================================
freq = duk(1);
valv = duk(2);
Pmanifold = duk(3);
P_reser = duk(4);
% =========================================================================
% Define os estados 
% =========================================================================
h0_anular =     xmk(1);
pressao_choke = xmk(2);
vazao_media =   xmk(3);

% =========================================================================
% Vari�veis calculadas 
% =========================================================================
visc = Par.visc;
pressao_fundo = Par.Vol_1.rho_1*Par.g*(Par.Vol_1.h_r-Par.Vol_1.h_p)+Par.Vol_1.rho_1*Par.g*h0_anular;
vazao_choke = ((valv).*((pressao_choke-Pmanifold).^0.5))*Par.k_choke;
vazao_reservatorio = (Par.IP*(P_reser - pressao_fundo));

% =========================================================================
% Define a fric��o, Deltap da bomba e corre��o da viscosidade
% =========================================================================
    Re_1 = (4*Par.Vol_1.rho_1*vazao_media)/(0.219*pi*visc);
    Re_2 = (4*Par.Vol_2.rho_2*vazao_media)/(0.219*pi*visc);
if Re_1<4000
    fric_1 = 64/Re_1;
    fric_2 = 64/Re_2;
else
    fric_1 = 0.36*Re_1^(-0.25);
    fric_2 = 0.36*Re_2^(-0.25);
end
    F_1 = (fric_1*vazao_media^2*Par.Vol_1.rho_1)/(2*pi*Par.Vol_1.r1^3);
    F_2 = (fric_2*vazao_media^2*Par.Vol_2.rho_2)/(2*pi*Par.Vol_2.r2^3);


H_aguabep = 4.330800000000000e+02; %ft
Q_aguabep = 4.401900000000000e+02; %bpd
y = -112.1374+6.6504*log(H_aguabep)+12.8429*log(Q_aguabep);
Q = exp((39.5276+26.5605*log(visc*1000)-y)/51.6565); %Pa.s to Cp;
Cq = (1.0-4.0327e-03*Q-1.7240e-04*Q^2);
CH = 1.0-4.4723e-03*Q -4.1800e-05*Q^2; % 80%
q_0a = vazao_media/Cq*(Par.f0/freq);
H_0 =  Par.Head(1)*q_0a^4 +  Par.Head(2)*q_0a^3 +  Par.Head(3)*q_0a^2 + Par.Head(4)*q_0a + Par.Head(5);
H = (CH*H_0*(freq/Par.f0)^2);
dp = H*Par.Vol_1.rho_1*Par.g;                     
% =========================================================================
% formula��o da EDO  
% =========================================================================
dydt(1,1) = (vazao_reservatorio - vazao_media)/Par.Anular;
dydt(2,1) = (Par.Vol_2.B2/Par.Vol_2.V2)*(vazao_media-vazao_choke);
dydt(3,1) = (Par.A_bar/(Par.rho_bar*Par.l_bar))*(pressao_fundo - pressao_choke - F_1 - F_2 - ...
            (Par.Vol_1.rho_1*Par.g*(Par.Vol_1.h_r-Par.Vol_1.h_p))-(Par.Vol_2.rho_2*Par.g*(Par.Vol_1.h_p-Par.Vol_2.h_c))+ (dp));
        
        
        
