% =========================================================================
%  Dados experimentais do BCS LEA   
% =========================================================================
function Cexp = dados_LEA_Exp(Cexp)

load('Dados_BCSLEA_20210818.mat');                                           % Carrega os dados experimentais 
Ts = 100;
Tinicio = 1;                                                                 % Seleciona o tempo inicial
Tfim = length(LEA_BCS_Instrumentacao);
LEA_BCS_Instrumentacao = LEA_BCS_Instrumentacao(Tinicio:Ts:Tfim,1:39);
Cexp.tempo_coleta = 1:length(LEA_BCS_Instrumentacao);                        % Constrói um vetor de tempo equivalente ao tempo especificado
Cexp.tempo = Cexp.tempo_coleta(end);                                         % [s] Tempo de simulação 

% =========================================================================
%  Dados experimentais da instrumentação   
% =========================================================================
Cexp.Ts = Ts;
Cexp.pressao_choke = double(((LEA_BCS_Instrumentacao(:,8))));                     % [Bar]  Pressao na cabeça do poço
Cexp.valvula_eletrica = double(((LEA_BCS_Instrumentacao(:,9))));                  % [%]    Abertura da válvula elétrica de controle na cabeça do poço
Cexp.valvula_pneumatica_topo = double(((LEA_BCS_Instrumentacao(:,10))));          % [%]    Valor da válvula pneumática de controle na cabeça do poço
Cexp.pressao_manifold = double((LEA_BCS_Instrumentacao(:,11)));                   % [Bar]  Pressao no manifold antes do medidor de vazão
Cexp.temperatura_manifold = LEA_BCS_Instrumentacao(:,12);                         % [ºC]   Temperatura de intake sensor de fundo BCS   
Cexp.vazao = double(LEA_BCS_Instrumentacao(:,13));                                % [m³/h] Vazão na choke
Cexp.pressao_manifold_coriolis = double((LEA_BCS_Instrumentacao(:,14)));          % [Bar]  Pressao no manifold após o coriolis
Cexp.pressao_reservatorio = double((LEA_BCS_Instrumentacao(:,15)));               % [Bar]  Pressao de reservatorio
Cexp.valvula_pneumatica_reservatorio = double(((LEA_BCS_Instrumentacao(:,16))));  % [%]    Valor da válvula pneumática de controle no reservatório
Cexp.pressao_fundo = double(LEA_BCS_Instrumentacao(:,17));                        % [Bar]  Pressao de fundo 
Cexp.temperatura_fundo = double(LEA_BCS_Instrumentacao(:,18));                    % [ºC]   Temperatura de reservatorio
Cexp.nivel_fundo = double(LEA_BCS_Instrumentacao(:,19));                          % [m]    Nível calculado em função da pressão de fundo (está no CLP)
Cexp.nivel_intake = double(LEA_BCS_Instrumentacao(:,19));                         % [m]    Nível calculado em função da pressão de fundo (está no CLP)
Cexp.vazao_reservatorio = double(LEA_BCS_Instrumentacao(:,20));                   % [m³/h] Vazão de reservatório
Cexp.torque_motor = double(LEA_BCS_Instrumentacao(:,21));                         % [%]    Torque do motor do BCS em % do valor máximo (consultar manual)
Cexp.rotacao = double(LEA_BCS_Instrumentacao(:,22));                              % [RPM]  Rotação do motor do BCS
Cexp.frequencia = double(LEA_BCS_Instrumentacao(:,23));                           % [Hz]   Frequência de operação do motor do BCS
Cexp.corrente_A = double(LEA_BCS_Instrumentacao(:,24));                           % [A]    Corrente da fase A do motor do BCS
Cexp.corrente_B = double(LEA_BCS_Instrumentacao(:,25));                           % [A]    Corrente da fase B do motor do BCS
Cexp.corrente_C = double(LEA_BCS_Instrumentacao(:,26));                           % [A]    Corrente da fase C do motor do BCS
Cexp.tensao_saida_inversor = double(LEA_BCS_Instrumentacao(:,27));                % [V]    Tensão de saída do inversor de frequência
Cexp.tensao_A = double(LEA_BCS_Instrumentacao(:,28));                             % [V]    Tensão da fase A do inversor de frequência
Cexp.tensao_B = double(LEA_BCS_Instrumentacao(:,29));                             % [V]    Tensão da fase A do inversor de frequência
Cexp.tensao_C = double(LEA_BCS_Instrumentacao(:,30));                             % [V]    Tensão da fase A do inversor de frequência
Cexp.referencia_frequencia_inversor = double(LEA_BCS_Instrumentacao(:,31));       % [V]    Referência de frequência escrita no inversor de frequência
Cexp.pressao_intake = double((LEA_BCS_Instrumentacao(:,32)))-1-4.3*855*9.81/10^5; % [Bar]  Pressao de intake da bomba, colhido no inversor, há duas correções necessárias: 1 - A pressão deve ser subtraída da atmosférica; 2- é necessário corrigir a posição da entrada da bomba, portanto retirar o peso da coluna de líquido.
Cexp.temperatura_intake = double((LEA_BCS_Instrumentacao(:,33)));                 % [ºC]   Temperatura de intake da bomba, colhido no inversor
Cexp.temperatura_motor = double((LEA_BCS_Instrumentacao(:,34)));                  % [ºC]   Temperatura do motor da bomba, colhido no inversor
Cexp.nivel_intake_Baker = ((Cexp.pressao_intake).*10.^5)./(9.81*855);             % [m]    Nível calculado com base na pressão de intake
Cexp.vazao_coriolis = double((LEA_BCS_Instrumentacao(:,36)));                    % [kg/s]  vazão do reservatorio coriolis
Cexp.densidade_coriolis = double((LEA_BCS_Instrumentacao(:,37)));                 % [kg/m³] Densidade do coriolis obtida da vazão de reservátorio
Cexp.temperatura_coriolis = double((LEA_BCS_Instrumentacao(:,38)));               % [ºC]    Temperatura do coriolis obtida da vazão de reservátorio




   