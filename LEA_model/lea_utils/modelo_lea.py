from lea_utils import dados_exp_LEA, lim_norm
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
        
from lea_utils import norm_values
from casadi import MX,Function
from casadi import vertcat as csvertcat
from casadi import sum1,nlpsol,integrator

# from lea_utils import mod
from lea_utils import model as ED

  
class ModeloLEA(object):
    def __init__(self,xss):
        self.xc,self.x0=norm_values()
        xc,x0=norm_values()
        self.xssn = (xss-x0)/xc
        self.nx = 3; self.nu = 4;
        self.x = MX.sym("x",self.nx); # Estados normalizados
        self.u = MX.sym("u",self.nu); # Entradas Exogenas
        dudt_max = MX.sym("dudt_max",2); # Taxa das entradas exogenas 
        dfq_max = 0.5;    # m�xima varia��o em f/s
        dzc_max = 1;      # máxima variação em zc
        ## dxdt simbolicos
        dx1,dx2,dx3,pin,H=ED.EDO(self.x,self.u)
        dxdt = csvertcat(dx1,dx2,dx3) 
        # dxdt = casadi.vertcat(dpbhdt,dpwhdt,dqdt) 
        self.Eq_Estado = Function('Eq_Estado',[self.x,self.u],[dxdt],
                            ['x','u'],['dxdt'])     
        #ny = y.size1()
        # Equações algébricas
        self.sea_nl = Function('sea_nl',[self.x,self.u],[pin,H],
                        ['x','u'],['pin','H']); # Sistema de Eq. Algebricas variaveis de sa�da 
    
    
    def regime_estacionario(self):
        #%% Calculo do estacionario
        #% Func��o objetivo
        dxdt_0 = self.Eq_Estado(self.x, self.u);
        J = sum1(dxdt_0**2);
        nx=self.nx
        #% Otimizador
        opt={
            'ipopt':{
                'print_level':0,
                'acceptable_tol':1e-8,
                'acceptable_obj_change_tol':1e-6,
                'max_iter':150
                },
            'print_time':0,
            }

        opt['ipopt']['print_level']=0;# %0,3
        opt['print_time']=0;
        opt['ipopt']['acceptable_tol']=1e-8;
        opt['ipopt']['acceptable_obj_change_tol']=1e-6;
        opt['ipopt']['max_iter']=50;
        #Encontrar o x que minimize J, dado u
        MMQ = {'x':self.x, 'f':J, 'p':self.u} # variáveis, função custo, entradas
        solver = nlpsol('solver', 'ipopt', MMQ, opt)
        # Restrições das variaveis de decis�o
        # minimo e máximo
        args={'lbx': np.zeros((nx,1)),'ubx':np.full((nx, 1), np.inf)}

        # Solução do otimizador
        sol=solver(x0=self.x, lbx=args['lbx'], ubx=args['ubx'], p=self.u);
        yss=self.sea_nl(sol['x'],self.u)
        Estacionario = Function('Estacionario',[self.x,self.u],[sol['x']],['x0','uss'],['xss']);
        return Estacionario
    def PredictionModel(self,ts):
        #% Parametros
        sedo = {'x': self.x, # Estados
                'p': self.u, #Variáveis exogenas
                'ode': self.Eq_Estado(self.x,self.u) # SEDO (Gerado no bcs_settings)
                }
        #% Criando o objeto p,ra integração da Eq_estado
        opt = {'tf':ts,'t0':0}   #% opções do integrador
        int_odes = integrator('int_odes','cvodes',sedo,opt);
        # objeto integrador
        res = int_odes(x0=self.x,p=self.u); #solução um passo a frente
        # Criacao do objeto para simulacao do BCS Eq de estado + Eq de Medicao
        Modelo_Predicao = Function('Modelo_Predicao',[self.x,self.u],[res['xf'].T],['xk_1','uk_1'],['xk'])
        return Modelo_Predicao





