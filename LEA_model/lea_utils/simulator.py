# from lea_utils import dados_exp_LEA, lim_norm
# from matplotlib import pyplot as plt
import numpy as np
   
from lea_utils import norm_values
from casadi import MX,Function
from casadi import vertcat as csvertcat
from casadi import sum1,nlpsol,integrator

# from lea_utils import mod
from lea_utils import model as ED
from lea_utils import PlotLEA,MLEA

  
class SimulatorLEA(object):
    def __init__(self,xss=np.array([[1.00000000e+05, 2.00000000e+05, 2.77777778e-05]])):
        self.xc,self.x0=norm_values()
        self.plotLEA = PlotLEA() 
        self.A=20
        self.BCS_EXP=None
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

    def getLEAdata(self,file_str,intervalo):
        self.BCS_EXP= MLEA(file_str,intervalo)
        print(f" File loaded {file_str}")   
    
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





