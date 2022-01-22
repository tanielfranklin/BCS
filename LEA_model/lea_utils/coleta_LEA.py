# =========================================================================
#  Ler dados experimentais do BCS LEA   
# =========================================================================
import scipy.io
import numpy as np
import math

def dados_exp_LEA(file_path,interval=np.array([0,np.inf])):
    #local do arquivo .mat do matlab coletado pelo experimento
    #interval = define o subdominio de dados experimentais em horas a ser usado
    
    mat = scipy.io.loadmat(file_path) # Carrega os dados experimentais                                       
    Ts = 100;
    Tfim=len(mat['LEA_BCS_Instrumentacao'])
    Tinicio=0
    
    if interval[1]!=np.inf:
        Tinicio = interval[0];# 1h 
        Tfim=interval[1] #4h
    #mat['LEA_BCS_Instrumentacao'].shape
    print(math.ceil(Tinicio / 3600))
    LEA_data = mat['LEA_BCS_Instrumentacao'][np.arange(Tinicio,Tfim,Ts),:]
    LEA={'tempo_coleta':np.arange(0,len(LEA_data))} # Constr�i um vetor de tempo equivalente ao tempo especificado
    LEA['tempo']=LEA['tempo_coleta'][-1]  # [s] Tempo de simula��o 
    

    # # =========================================================================
    # #  Dados experimentais da instrumenta��o   
    # # =========================================================================
    dados={'Ts':Ts,
    'pressao_choke':np.float32(((LEA_data[:,7]))),                     # [Bar]  Pressao na cabe�a do po�o
    'valvula_eletrica':np.float32(((LEA_data[:,8]))),                  # [#]    Abertura da v�lvula el�trica de controle na cabe�a do po�o
    'valvula_pneumatica_topo': np.float32(((LEA_data[:,9]))),          # [#]    Valor da v�lvula pneum�tica de controle na cabe�a do po�o
    'pressao_manifold': np.float32((LEA_data[:,10])),                   # [Bar]  Pressao no manifold antes do medidor de vaz�o
    'temperatura_manifold': LEA_data[:,11],                         # [�C]   Temperatura de intake sensor de fundo BCS   
    'vazao' : np.float32(LEA_data[:,12]),                                # [m�/h] Vaz�o na choke
    'pressao_manifold_coriolis' : np.float32((LEA_data[:,13])),          # [Bar]  Pressao no manifold ap�s o coriolis
    'pressao_reservatorio' : np.float32((LEA_data[:,14])),               # [Bar]  Pressao de reservatorio
    'valvula_pneumatica_reservatorio' : np.float32(((LEA_data[:,15]))),  # [#]    Valor da v�lvula pneum�tica de controle no reservat�rio
    'pressao_fundo' : np.float32(LEA_data[:,16]),                        # [Bar]  Pressao de fundo 
    'temperatura_fundo' : np.float32(LEA_data[:,17]),                    # [�C]   Temperatura de reservatorio
    'nivel_fundo' : np.float32(LEA_data[:,18]),                          # [m]    N�vel calculado em fun��o da press�o de fundo (est� no CLP)
    'nivel_intake' : np.float32(LEA_data[:,18]),                         # [m]    N�vel calculado em fun��o da press�o de fundo (est� no CLP)
    'vazao_reservatorio' : np.float32(LEA_data[:,19]),                   # [m�/h] Vaz�o de reservat�rio
    'torque_motor' : np.float32(LEA_data[:,20]),                         # [#]    Torque do motor do BCS em # do valor m�ximo (consultar manual)
    'rotacao' : np.float32(LEA_data[:,21]),                              # [RPM]  Rota��o do motor do BCS
    'frequencia' : np.float32(LEA_data[:,22]),                           # [Hz]   Frequ�ncia de opera��o do motor do BCS
    'corrente_A' : np.float32(LEA_data[:,23]),                           # [A]    Corrente da fase A do motor do BCS
    'corrente_B' : np.float32(LEA_data[:,24]),                           # [A]    Corrente da fase B do motor do BCS
    'corrente_C' : np.float32(LEA_data[:,25]),                           # [A]    Corrente da fase C do motor do BCS
    'tensao_saida_inversor' : np.float32(LEA_data[:,26]),                # [V]    Tens�o de sa�da do inversor de frequ�ncia
    'tensao_A' : np.float32(LEA_data[:,27]),                             # [V]    Tens�o da fase A do inversor de frequ�ncia
    'tensao_B' : np.float32(LEA_data[:,28]),                             # [V]    Tens�o da fase A do inversor de frequ�ncia
    'tensao_C' : np.float32(LEA_data[:,29]),                             # [V]    Tens�o da fase A do inversor de frequ�ncia
    'referencia_frequencia_inversor' : np.float32(LEA_data[:,30])*0.1,       # [V]    Refer�ncia de frequ�ncia escrita no inversor de frequ�ncia
    'pressao_intake' : np.float32((LEA_data[:,31]))-1-4.3*855*9.81/10**5, # [Bar]  Pressao de intake da bomba, colhido no inversor, h� duas corre��es necess�rias: 1 - A press�o deve ser subtra�da da atmosf�rica, 2- � necess�rio corrigir a posi��o da entrada da bomba, portanto retirar o peso da coluna de l�quido.
    'temperatura_intake' : np.float32((LEA_data[:,32])),                 # [�C]   Temperatura de intake da bomba, colhido no inversor
    'temperatura_motor' : np.float32((LEA_data[:,33])),                  # [�C]   Temperatura do motor da bomba, colhido no inversor
    
    'vazao_coriolis' : np.float32((LEA_data[:,35])),                    # [kg/s]  vaz�o do reservatorio coriolis
    'densidade_coriolis' : np.float32((LEA_data[:,36])),                 # [kg/m�] Densidade do coriolis obtida da vaz�o de reserv�torio
    'temperatura_coriolis' : np.float32((LEA_data[:,37]))                  # [�C]    Temperatura do coriolis obtida da vaz�o de reserv�torio
    }
    LEA.update(dados)
    LEA['nivel_intake_Baker']=(LEA['pressao_intake']*10**5)/(9.81*855)           # [m]    N�vel calculado com base na press�o de intake              
    return LEA
# LEA=dados_LEA_Exp()
# print(LEA['valvula_pneumatica_topo'])