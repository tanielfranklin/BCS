import numpy as np
from matplotlib import pyplot as plt

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
            ax1.set_xticklabels([])
        plt.grid(True)

def APRBS(a_range,b_range,nstep):
    # random signal generation
    a = np.random.rand(nstep) * (a_range[1]-a_range[0]) + a_range[0] # range for amplitude
    b = np.random.rand(nstep) *(b_range[1]-b_range[0]) + b_range[0] # range for frequency
    b = np.round(b)
    b = b.astype(int)

    b[0] = 0

    for i in range(1,np.size(b)):
        b[i] = b[i-1]+b[i]

    # Random Signal
    i=0
    random_signal = np.zeros(nstep)
    while b[i]<np.size(random_signal):
        k = b[i]
        random_signal[k:] = a[i]
        i=i+1

    # PRBS
    a = np.zeros(nstep)
    j = 0
    while j < nstep:
        a[j] = 5
        a[j+1] = -5
        j = j+2

    i=0
    prbs = np.zeros(nstep)
    while b[i]<np.size(prbs):
        k = b[i]
        prbs[k:] = a[i]
        i=i+1
    return [random_signal, prbs]