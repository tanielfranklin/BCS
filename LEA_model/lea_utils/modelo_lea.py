from lea_utils import dados_exp_LEA
from matplotlib import pyplot as plt
import numpy as np

class MLEA(object):
    def __init__(self,file_data_exp,intervalo_horas):        
        self.LEA=dados_exp_LEA(file_data_exp,intervalo_horas)
        self.ts=self.LEA['Ts']
        self.nsim=self.LEA['tempo']+1
        
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
    

    def plot_exogenous(self):
        label = ['f(Hz)','z(%)','Pman(bar)','Pr(bar)'];
        tempo,exo,_,_= self.input_output_data()
        exo[2]=exo[2]/1e5
        exo[3]=exo[3]/1e5
        fig3=plt.figure()
        for i,var in enumerate(exo):
            ax1=fig3.add_subplot(len(label),1,i+1)
            ax1.plot(tempo ,var, label=label[i])
            # ax1.plot(tempo_hora ,output_signal/1e5, ':r')
            ax1.set_ylabel(label[i])
            if i+1!=len(exo):
                ax1.set_xticklabels([])
            plt.grid(True)
        return fig3
    
    def plot_states(self):
        label = ['Pbh(bar)','Pwh(bar)','q(m3/h)'];
        tempo,_,st,_= self.input_output_data()
        ### Set Enginneering dimensions###########
        x_set_dim=[1/1e5,1/1e5,3600]
        var=[a * b for a, b in zip(st,x_set_dim )]
        # for i,j in zip(st,x_set_dim ):
        #     var.append(i*j)   
        ###########################################

        fig3=plt.figure()
        for i,var in enumerate(var):
            ax1=fig3.add_subplot(len(label),1,i+1)
            ax1.plot(tempo ,var, label=label[i])
            ax1.set_ylabel(label[i])
            if i+1!=len(st):
                ax1.set_xticklabels([])
            plt.grid(True)
        return fig3
        
    







