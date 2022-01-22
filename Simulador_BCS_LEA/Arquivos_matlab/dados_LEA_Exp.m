% =========================================================================
%  Dados experimentais do BCS LEA   
% =========================================================================
function Cexp = dados_LEA_Exp(Cexp)

load('Dados_BCSLEA_20210818.mat');                                           % Carrega os dados experimentais 
Ts = 100;
Tinicio = 1;                                                                 % Seleciona o tempo inicial
Tfim = length(LEA_BCS_Instrumentacao);
LEA_BCS_Instrumentacao = LEA_BCS_Instrumentacao(Tinicio:Ts:Tfim,1:39);
Cexp.tempo_coleta = 1:length(LEA_BCS_Instrumentacao);                        % Constr�i um vetor de tempo equivalente ao tempo especificado
Cexp.tempo = Cexp.tempo_coleta(end);                                         % [s] Tempo de simula��o 

% =========================================================================
%  Dados experimentais da instrumenta��o   
% =========================================================================
Cexp.Ts = Ts;
Cexp.pressao_choke = double(((LEA_BCS_Instrumentacao(:,8))));                     % [Bar]  Pressao na cabe�a do po�o
Cexp.valvula_eletrica = double(((LEA_BCS_Instrumentacao(:,9))));                  % [%]    Abertura da v�lvula el�trica de controle na cabe�a do po�o
Cexp.valvula_pneumatica_topo = double(((LEA_BCS_Instrumentacao(:,10))));          % [%]    Valor da v�lvula pneum�tica de controle na cabe�a do po�o
Cexp.pressao_manifold = double((LEA_BCS_Instrumentacao(:,11)));                   % [Bar]  Pressao no manifold antes do medidor de vaz�o
Cexp.temperatura_manifold = LEA_BCS_Instrumentacao(:,12);                         % [�C]   Temperatura de intake sensor de fundo BCS   
Cexp.vazao = double(LEA_BCS_Instrumentacao(:,13));                                % [m�/h] Vaz�o na choke
Cexp.pressao_manifold_coriolis = double((LEA_BCS_Instrumentacao(:,14)));          % [Bar]  Pressao no manifold ap�s o coriolis
Cexp.pressao_reservatorio = double((LEA_BCS_Instrumentacao(:,15)));               % [Bar]  Pressao de reservatorio
Cexp.valvula_pneumatica_reservatorio = double(((LEA_BCS_Instrumentacao(:,16))));  % [%]    Valor da v�lvula pneum�tica de controle no reservat�rio
Cexp.pressao_fundo = double(LEA_BCS_Instrumentacao(:,17));                        % [Bar]  Pressao de fundo 
Cexp.temperatura_fundo = double(LEA_BCS_Instrumentacao(:,18));                    % [�C]   Temperatura de reservatorio
Cexp.nivel_fundo = double(LEA_BCS_Instrumentacao(:,19));                          % [m]    N�vel calculado em fun��o da press�o de fundo (est� no CLP)
Cexp.nivel_intake = double(LEA_BCS_Instrumentacao(:,19));                         % [m]    N�vel calculado em fun��o da press�o de fundo (est� no CLP)
Cexp.vazao_reservatorio = double(LEA_BCS_Instrumentacao(:,20));                   % [m�/h] Vaz�o de reservat�rio
Cexp.torque_motor = double(LEA_BCS_Instrumentacao(:,21));                         % [%]    Torque do motor do BCS em % do valor m�ximo (consultar manual)
Cexp.rotacao = double(LEA_BCS_Instrumentacao(:,22));                              % [RPM]  Rota��o do motor do BCS
Cexp.frequencia = double(LEA_BCS_Instrumentacao(:,23));                           % [Hz]   Frequ�ncia de opera��o do motor do BCS
Cexp.corrente_A = double(LEA_BCS_Instrumentacao(:,24));                           % [A]    Corrente da fase A do motor do BCS
Cexp.corrente_B = double(LEA_BCS_Instrumentacao(:,25));                           % [A]    Corrente da fase B do motor do BCS
Cexp.corrente_C = double(LEA_BCS_Instrumentacao(:,26));                           % [A]    Corrente da fase C do motor do BCS
Cexp.tensao_saida_inversor = double(LEA_BCS_Instrumentacao(:,27));                % [V]    Tens�o de sa�da do inversor de frequ�ncia
Cexp.tensao_A = double(LEA_BCS_Instrumentacao(:,28));                             % [V]    Tens�o da fase A do inversor de frequ�ncia
Cexp.tensao_B = double(LEA_BCS_Instrumentacao(:,29));                             % [V]    Tens�o da fase A do inversor de frequ�ncia
Cexp.tensao_C = double(LEA_BCS_Instrumentacao(:,30));                             % [V]    Tens�o da fase A do inversor de frequ�ncia
Cexp.referencia_frequencia_inversor = double(LEA_BCS_Instrumentacao(:,31));       % [V]    Refer�ncia de frequ�ncia escrita no inversor de frequ�ncia
Cexp.pressao_intake = double((LEA_BCS_Instrumentacao(:,32)))-1-4.3*855*9.81/10^5; % [Bar]  Pressao de intake da bomba, colhido no inversor, h� duas corre��es necess�rias: 1 - A press�o deve ser subtra�da da atmosf�rica; 2- � necess�rio corrigir a posi��o da entrada da bomba, portanto retirar o peso da coluna de l�quido.
Cexp.temperatura_intake = double((LEA_BCS_Instrumentacao(:,33)));                 % [�C]   Temperatura de intake da bomba, colhido no inversor
Cexp.temperatura_motor = double((LEA_BCS_Instrumentacao(:,34)));                  % [�C]   Temperatura do motor da bomba, colhido no inversor
Cexp.nivel_intake_Baker = ((Cexp.pressao_intake).*10.^5)./(9.81*855);             % [m]    N�vel calculado com base na press�o de intake
Cexp.vazao_coriolis = double((LEA_BCS_Instrumentacao(:,36)));                    % [kg/s]  vaz�o do reservatorio coriolis
Cexp.densidade_coriolis = double((LEA_BCS_Instrumentacao(:,37)));                 % [kg/m�] Densidade do coriolis obtida da vaz�o de reserv�torio
Cexp.temperatura_coriolis = double((LEA_BCS_Instrumentacao(:,38)));               % [�C]    Temperatura do coriolis obtida da vaz�o de reserv�torio




   