from matplotlib import pyplot as plt  

class PlotLEA(object):
    def __init__(self,ss_label=None,u_label=None,Experimental=None,Sim_result=None):
        self.ss_label=['Pbh(bar)','Pwh(bar)','q(m3/h)'];
        self.u_label=['f(Hz)','z(%)','Pman(bar)','Pr(bar)'];
        self.BCS=Experimental
        self.sim=Sim_result
    
    def update_label_ss(self,label_str):
        #string list with three elements
        self.ss_label=label_str

    def update_label_u(self,label_str):
        #string list with four elements
        self.ss_label=label_str

    def plot_states(self,x,tempo):                
        ### Set Enginneering dimensions###########
        x_set_dim=[1/1e5,1/1e5,3600]
        var=[a * b for a, b in zip(x,x_set_dim )]
        # for i,j in zip(st,x_set_dim ):
        #     var.append(i*j)   
        ###########################################
        fig3=plt.figure()
        for i,var in enumerate(var):
            ax1=fig3.add_subplot(len(self.ss_label),1,i+1)
            ax1.plot(tempo ,var, label=self.ss_label[i])
            ax1.set_ylabel(self.ss_label[i])
            if i+1!=len(x):
                ax1.set_xticklabels([])
            plt.grid(True)
        return fig3

    def plot_exogenous(self,exo,tempo):
        exo[2]=exo[2]/1e5
        exo[3]=exo[3]/1e5
        fig3=plt.figure()
        for i,var in enumerate(exo):
            ax1=fig3.add_subplot(len(self.u_label),1,i+1)
            ax1.plot(tempo ,var, label=self.u_label[i])
            # ax1.plot(tempo_hora ,output_signal/1e5, ':r')
            ax1.set_ylabel(self.u_label[i])
            if i+1!=len(exo):
                ax1.set_xticklabels([])
            plt.grid(True)
        return fig3    

