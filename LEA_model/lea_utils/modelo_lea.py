from lea_utils import dados_exp_LEA, PlotLEA
from matplotlib import pyplot as plt
import numpy as np

class MLEA(object):
    def __init__(self,file_data_exp,intervalo_horas):        
        self.LEA=dados_exp_LEA(file_data_exp,intervalo_horas)
        self.ts=self.LEA['Ts']
        self.nsim=self.LEA['tempo']+1
        self.x=self.get_ss_values()
        self.u=self.get_exo_values()
        self.tempo_hora = np.arange(0,self.nsim*self.ts,self.ts)/3600
        self.plotlea=PlotLEA()
        self.plot_ss=self.plotlea.plot_states(self.x,self.tempo_hora)
        self.plot_u=self.plotlea.plot_exogenous(self.u,self.tempo_hora)
        plt.close(self.plot_ss)
        plt.close(self.plot_u)
        
    def get_exo_values(self):
        #Perturbações#######################################
        pman=self.LEA['pressao_manifold_coriolis']*1e5 
        pres=self.LEA['pressao_reservatorio']*1e5  #SI
        #Entradas exógenas manipuladas######################
        zc = self.LEA['valvula_pneumatica_topo']#[0:nsim]
        fk = self.LEA['referencia_frequencia_inversor']#[0:nsim]
        u=[fk,zc,pman,pres]
        return u
    def get_ss_values(self):
        #Estados############################################
        q=self.LEA['vazao']/3600 #SI
        pbh=self.LEA['pressao_fundo']*1e5 #SI
        pwh=self.LEA['pressao_choke']*1e5 #SI
        x=[pbh,pwh,q]
        return x
   
    def input_output_data(self):
        u=self.get_exo_values()
        x=self.get_ss_values()
        pin_exp=self.LEA['pressao_intake']*1e5
        y=[pin_exp]
        tempo=self.LEA['pressao_intake']
        nsim=self.LEA['tempo']+1
        tempo_hora = np.arange(0,nsim*self.ts,self.ts)/3600
        return tempo_hora,u,x,y
    


    

        
