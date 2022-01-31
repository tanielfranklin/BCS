from matplotlib import pyplot as plt


class PlotLEA(object):
    def __init__(self,ss_label=None,u_label=None,Experimental=None,Sim_result=None):
        self.ss_label=['Pbh(bar)','Pwh(bar)','q(m3/h)'];
        self.u_label=['f(Hz)','z(%)','Pman(bar)','Pr(bar)'];
        self.y_label = [r'$p_{in}(bar)$','H(m)'];
        self.u_scale=[1,1,1/1e5,1/1e5]
        self.ss_scale=[1/1e5,1/1e5,3600]
        self.BCS=Experimental
        self.sim=Sim_result
    
    def update_label_ss(self,label_str):
        #string list with three elements
        self.ss_label=label_str
    def update_label_y(self,label_str):
        #string list with two elements
        self.y_label=label_str
    def update_label_u(self,label_str):
        #string list with four elements
        self.u_label=label_str
    def update_scale_u(self,scale_list):
        #string list with four elements
        self.u_scale=scale_list
    def update_scale_ss(self,label_list):
        #string list with three elements
        self.ss_scale=label_list

    def plot_states(self,data,tempo):                
        ### Set Enginneering dimensions###########
        
        if len(data)==1:
            var=[a * b for a, b in zip(data[0],self.ss_scale )]
        elif len(data)==2:
            var=[a * b for a, b in zip(data[0],self.ss_scale)]
            var_exp=[a * b for a, b in zip(data[1],self.ss_scale)]
        else:
            raise ValueError("Invalid arguments for input:x (only len<=2")
        ###########################################
        fig3=plt.figure()
        for i,a in enumerate(var):
            ax1=fig3.add_subplot(len(self.ss_label),1,i+1)
            
            ax1.plot(tempo ,a,"-k", label="simulated")
            if len(data)==2:
                
                ax1.plot(tempo ,var_exp[i],":b", label="ground true")
            ax1.set_ylabel(self.ss_label[i])
            if i+1!=len(data[0]):
                ax1.set_xticklabels([])
            ax1.set_xlabel("Time(h)")
            plt.grid(True)
        plt.legend(bbox_to_anchor=(1, 3.8), ncol = 2)
        return fig3

    def plot_exogenous(self,exo,tempo):
        var_exo=[a * b for a, b in zip(exo,self.u_scale)]
        fig3=plt.figure()
        for i,var in enumerate(var_exo):
            ax1=fig3.add_subplot(len(self.u_label),1,i+1)
            ax1.plot(tempo ,var, label=self.u_label[i])
            # ax1.plot(tempo_hora ,output_signal/1e5, ':r')
            ax1.set_ylabel(self.u_label[i])
            if i+1!=len(exo):
                ax1.set_xticklabels([])
            plt.grid(True)
        return fig3

    def plot_y(self,tempo,y):
        
        fig1=plt.figure()
        scale=[1/1e5,1]
        if len(y)==1:
            var=[i*j for i,j in zip(y[0],scale)]
        elif len(y)==2:
            var=[i*j for i,j in zip(y[0],scale)]
            var_exp=[i*j for i,j in zip(y[1],scale)]
        else:
            raise ValueError("Invalid arguments for input:x (only len<=2")
        
        for i,a in enumerate(var):
            ax = fig1.add_subplot(len(self.y_label),1,i+1)
            ax.plot(tempo,a,"-k", label='Simulated')
            if len(y)==2:
                ax.plot(tempo,var_exp[i],":b", label='Ground True')
            ax.set_ylabel(self.y_label[i])
            #ax.set(xlim=(tempo[0], nsim*ts))
            # ax.set(ylim=(40,62))
            plt.grid(True)
        ax.legend();
        ax.set_xlabel('Time (h)')
        plt.legend(bbox_to_anchor=(1, 2.5), ncol = 2)
        
        return fig1
    

    # #exec(compile(open('envelope.py', "rb").read(), 'envelope.py', 'exec')) #% Roda arquivo com parâmetros do modelo BCS
# fig4,ax4=plt.subplots()
# plt.grid(True)
# #BCS['Envelope']['fig'](ax4); # grafico do envelope
# #
# # Evolução dentro do envelope
# ax4.plot(Xk[2,0:].T*3600,Yk[1,0:].T,'--k')
# ax4.plot(Xk[2,0]*3600,Yk[1,0],'o')#,'MarkerFaceColor',[0,1,0],'MarkerEdgeColor',[0,0,0])
# ax4.plot(Xk[2,-1]*3600,Yk[1,-1],'o')#,'MarkerFaceColor',[1,0,0],'MarkerEdgeColor',[0,0,0])
# ax4.annotate('t=0',
#              xy=(float(Xk[2,0]*3600),float(Yk[1,0])),
#              xytext=(float(Xk[2,0]*3600)-5,float(Yk[1,0])+10),
#              arrowprops=dict(facecolor='green', shrink=0.01))

# ax4.annotate('t='+str(nsim),
#              xy=(float(Xk[2,-1]*3600),float(Yk[1,-1])),
#              xytext=(float(Xk[2,-1]*3600)-7,float(Yk[1,-1])+10),
#              arrowprops=dict(facecolor='red', shrink=0.01))
# plt.show()
    

    
        

