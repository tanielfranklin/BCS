% =========================================================================
%  Par�metros do sistema LEA BCS   
% =========================================================================
function Par = dados_LEA_Par(Par)
% =========================================================================
% Volume 1
% =========================================================================
    Par.Vol_1.B1 = 1.800e+09;                                  % [Pa]     Compressibilidade
    Par.Vol_1.rho_1 = 836.8898;                                % [kg/m^3] Densidade do fluido obtida pela m�dia dos experimentos no coriolis          
    Par.Vol_1.r1 = 0.11;                                       % [m]      Raio do tubo 
    Par.Vol_1.l1 =9.3;                                         % [m]      Comprimento do reservatorio at� a intake
    Par.Vol_1.h_r = 32;                                        % [m]      Profundidade do reservat�rio
    Par.Vol_1.h_p = 22.7;                                      % [m]      Profundidade do conjunto BCS
    Par.Vol_1.V1 = Par.Vol_1.l1*pi*Par.Vol_1.r1^2;             % [m^3]    C�lculo do volume 
    
% =========================================================================
% Volume 2
% =========================================================================    
    Par.Vol_2.B2 = 1.800e+09;                                  % [Pa]     Compressibilidade
    Par.Vol_2.rho_2 = 836.8898;                                % [kg/m^3] Densidade do fluido obtida pela m�dia dos experimentos no coriolis          
    Par.Vol_2.r2 = 0.0375;                                     % [m]      Raio do tubo 
    Par.Vol_2.l2 = 22.7;                                       % [m]      Comprimento da intake at� a choke
    Par.Vol_2.h_c = 0;                                         % [m]      Profundidade do choke
    Par.Vol_2.V2 = Par.Vol_2.l2*pi*Par.Vol_2.r2^2;             % [m^3]    C�lculo do volume 
    
% =========================================================================
% outras
% =========================================================================    
    Par.Pman =    0.139365196228027*1e05;                      % [Pa] Press�o de Manifold 1 atm
    Par.Resv =    2.2000e+05;                                  % [Pa] M�ximo do valor experimental
    Par.k_choke = 3.10049373466703e-08;                    
    Par.visc =    0.012;                                       % [Pa.s] Viscosidade din�mica
    Par.g =       9.81;                                        % [m/s^2] Gravidade
    Par.IP =      2.7000e-08;                                  % [m^3/(s*Pa)]Indice de Produtividade
    Par.fMax = 60;                                             % [Hz]   Frequ�ncia m�xima admiss�vel
    Par.fMin = 30;                                             % [Hz]   Frequ�ncia m�nima admiss�vel
    Par.f0 =   60;                                             % Frequecia inicial
    Par.Anular =  0.033595;                                    % Area do anular LEA
    Par.I_np = 25;                                             % [A] corrente nominal  
    Par.P_np = 13422.5982;                                     % [W] ou 18 [hp] obtida do Q_bep
    Par.q0_min = 0.5/3600; % Vaz�o m�nima de refer�ncia
    Par.q0_max = 4/3600;   % Vaz�o m�xima de refer�ncia

    Par.l_bar = (Par.Vol_1.l1 + Par.Vol_2.l2)/2;
    Par.r_bar = ((Par.Vol_1.r1*Par.Vol_1.l1)+(Par.Vol_2.r2*Par.Vol_2.l2))/(Par.Vol_1.l1+Par.Vol_2.l2);
    Par.A_bar =  pi*Par.r_bar^2;
    Par.rho_bar = ((Par.Vol_1.rho_1*Par.Vol_1.V1)+(Par.Vol_2.rho_2*Par.Vol_2.V2))/(Par.Vol_1.V1+Par.Vol_2.V2);

% =========================================================================
% Curva da bomba
% ========================================================================= 
    Par.Pot  = [3.14;3.41;3.57;3.60]*745.7; % Hp to Watts
    Par.Head = [78255575038455.9;-243021891442.447;154711075.976357;-63654.8760768729;187.303058039876]; %corrigido fminsearch (11_04_2021)
    Par.Eff =  [-38000037638726.0,76885738063.1052,-104037865.052405,92277.4570774092,-0.000102311045885796];
    


   