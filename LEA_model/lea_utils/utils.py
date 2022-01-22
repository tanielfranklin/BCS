def plot_exogenous(tempo,exo):
    label = ['f(Hz)','z(%)','Pman(bar)','Pr(bar)'];
    exo[2]=exo[2]/1e5
    exo[3]=exo[3]/1e5
    fig3=plt.figure()
    for i,var in enumerate(exo):
        ax1=fig3.add_subplot(len(label),1,i+1)
        ax1.plot(tempo ,var, label=label[i])
        # ax1.plot(tempo_hora ,output_signal/1e5, ':r')
        ax1.set_ylabel(label[i])
        if i+1!=len(exo):
            print(i,len(exo))
            ax1.set_xticklabels([])
        plt.grid(True)