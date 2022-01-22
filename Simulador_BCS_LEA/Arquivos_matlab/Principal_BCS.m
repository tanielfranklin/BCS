% =========================================================================
% Projeto "Po�o BCS Inteligente" 
% Valida��o dos dados experimentais do BCS LEA  
% =========================================================================
    clear 
    close all hidden
    close all force
    clc
    Par =  dados_LEA_Par;                  % Par�metros do BCS LEA                               
    Cexp = dados_LEA_Exp;                  % Dados experimentais do BCS LEA   
    nsim = Cexp.tempo;                     % [s] Tempo de simula��o 
    Ts =   Cexp.Ts;                        % [s] Per�odo de amostragem
% =========================================================================
%  Define as condi��es iniciais do BCS LEA   
% =========================================================================
    h_anular =      Cexp.nivel_intake(1,1);        % [m] n�vel do anular             
    pressao_choke = Cexp.pressao_choke(1,1)*1e05;  % [Pa] press�o na choke    
    vazao_media =   Cexp.vazao(1,1)/3600;          % [m�/s] vaz�o      
    ypk = [h_anular pressao_choke vazao_media]'; 
    
% =========================================================================
%  Define as entradas do BCS LEA   
% =========================================================================
    freq = Cexp.referencia_frequencia_inversor(1:nsim)*0.1;                   % [Hz] frequencia de operacao
    val_pneumatica = Cexp.valvula_pneumatica_topo(1:nsim);                    % [%]  v�lvula pneum�tica
    pressao_reser =  Cexp.pressao_reservatorio(1:nsim)*1e05;                  % [Pa] pressao de reservatorio do LEA
    pressao_manifold = Cexp.pressao_manifold_coriolis(1:nsim)*1e05;           % [Pa] Press�o de manifold choke
    temperatura_fundo = Cexp.temperatura_fundo(1:nsim);                       % [�C] Temperatura de fundo
% =========================================================================
% Define os vetores para aloca��o de mem�ria   
% =========================================================================
    pressao_intake = zeros(1,length(nsim));    
    pressao_fundo = zeros(1,length(nsim));     
    vazao_choke = zeros(1,length(nsim));       
    vazao_reservatorio = zeros(1,length(nsim));
    ypk_planta = zeros(3,length(nsim));
% =========================================================================
% Dados experimentais   
% =========================================================================
    ypk_planta_Exp = [Cexp.nivel_intake(1:nsim) Cexp.pressao_choke(1:nsim) Cexp.vazao(1:nsim)]';

% =========================================================================
% Simula��o   
% =========================================================================
    h = waitbar(0,'Valida��o Experimental...');
    options = odeset('Abstol',1e-6,'Reltol',1e-6);

for k=1:1:nsim
        waitbar(k/nsim,h);    
        duk = [freq(k) val_pneumatica(k) pressao_manifold(k) pressao_reser(k) temperatura_fundo(k)]';
        ypk_planta(:,k) = ypk;
        [t,dydt]=ode15s(@dinamica_BCS,[0 Ts],ypk,options,duk,Par);
        h_anular = dydt(end,1);
        pressao_choke   = dydt(end,2);      
        vazao_media   = dydt(end,3);
        ypk = [h_anular pressao_choke vazao_media]';
        var_calculo = calculo_LEA(ypk,duk,Par);
        pressao_intake(:,k) =      var_calculo(1);            
        pressao_fundo(:,k) =       var_calculo(2);          
        vazao_choke(:,k) =         var_calculo(3);             
        vazao_reservatorio(:,k) =  var_calculo(4); 
end

% =========================================================================
% Monta gr�fico  
% =========================================================================
tempo_hora = (Cexp.Ts:Cexp.Ts:nsim*Cexp.Ts)/3600;
figure1 = figure('position',[185    30   915   610],'Color',[1 1 1]);
axes1 = axes('Parent',figure1);
%ylim(axes1,[0 max(Cexp.nivel_intake)+5])
%xlim(axes1,[0 Cexp.tempo])
hold(axes1,'all');
box(axes1,'on');
plot(tempo_hora,ypk_planta(1,:),'b','LineWidth',2)
hold on
plot(tempo_hora,ypk_planta_Exp(1,:),'k--','LineWidth',2)
title('N�vel do anular','FontSize',18)
xlabel('Tempo/h','FontSize',18)
ylabel('N�vel Anular/m','FontSize',18)
set(axes1,'FontSize',18,'FontWeight','bold');
legend({'Simulado','Experimental'},'FontSize',18)
set(legend,'Location','best');
grid

% =================================================================
% Press�o na cabe�a do po�o (Choke)  
% =================================================================
figure2 = figure('position',[185    30   915   610],'Color',[1 1 1]);
axes1 = axes('Parent',figure2);
%ylim(axes1,[1 max(Cexp.pressao_choke)+1])
%xlim(axes1,[0 Cexp.tempo])
hold(axes1,'all');
box(axes1,'on');
plot(tempo_hora,ypk_planta(2,:)/1e+05,'b','LineWidth',2)
hold on
plot(tempo_hora,ypk_planta_Exp(2,:),'k--','LineWidth',2)
title('Press�o na choke','FontSize',18)
ylabel('Press�o/bar','FontSize',18)
xlabel('Tempo/h','FontSize',18)
set(axes1,'FontSize',18,'FontWeight','bold');
legend({'Simulado','Experimental'},'FontSize',18)
set(legend,'Location','best');
grid

% =================================================================
% Vaz�o  
% =================================================================
figure3 = figure('position',[185    30   915   610],'Color',[1 1 1]);
axes1 = axes('Parent',figure3);
hold(axes1,'all');
box(axes1,'on');
%ylim(axes1,[0 max(Cexp.vazao)+1])
%xlim(axes1,[0 Cexp.tempo])
plot(tempo_hora,vazao_choke*3600,'b','LineWidth',2)
hold on
plot(tempo_hora,ypk_planta_Exp(3,:),'k--','LineWidth',2)
title('Vaz�o na choke','FontSize',18)
ylabel('Vaz�o/m�/h','FontSize',18)
xlabel('Tempo/h','FontSize',18)
set(axes1,'FontSize',18,'FontWeight','bold');
legend({'Simulado','Experimental'},'FontSize',18)
set(legend,'Location','best');
grid

% =================================================================
% Frequencia  
% =================================================================
figure5 = figure('position',[185    30   915   610],'Color',[1 1 1]);
axes1 = axes('Parent',figure5);
%ylim(axes1,[0 max(Cexp.valvula_pneumatica_topo)+1])
%xlim(axes1,[0 tempo_hora])
hold(axes1,'all');
box(axes1,'on');
plot(tempo_hora,freq,'b','LineWidth',2)
hold on
plot(tempo_hora,val_pneumatica,'k','LineWidth',2)
title('Frequ�ncia e V�lvula','FontSize',18)
xlabel('Tempo/h','FontSize',18)
ylabel({'Variavel manipulada'},'FontSize',18)
legend({'Frequ�ncia/Hz','Valvula/(%)'},'FontSize',18)
set(axes1,'FontSize',18,'FontWeight','bold');
set(legend,'Location','best');
grid

xo=[pressao_fundo(1);pressao_choke(1);vazao_choke(1)]
disp('FIM DO PROGRAMA');



