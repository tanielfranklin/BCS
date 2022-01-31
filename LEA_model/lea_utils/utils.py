import numpy as np
from matplotlib import pyplot as plt

def plot_states_double(tempo,ss,ss_exp):
    label = ['Pbh(bar)','Pwh(bar)','q(m3/h)'];
    ### Set Enginneering dimensions###########
    x_set_dim=[1/1e5,1/1e5,3600]
    var, var_exp=[],[]
    for i,j,k in zip(ss,x_set_dim,ss_exp ):
        print(j)
        var.append(i*j)   
        var_exp.append(j*k)
    ###########################################

    fig3=plt.figure()
    for i,(var_ss,var_ss_exp) in enumerate(zip(var,var_exp)):
        ax1=fig3.add_subplot(len(label),1,i+1)
        ax1.plot(tempo ,var_ss,'-b',label="sim")
        ax1.plot(tempo ,var_ss_exp, ':r', label="exp")
        ax1.set_ylabel(label[i])
        if i+1!=len(ss):
            ax1.set_xticklabels([])
        plt.grid(True)
    ax1.set_xlabel("Time(h)")
    plt.legend(bbox_to_anchor=(1.0, -0.3), ncol = 2)
    return fig3

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
    return random_signal


def split_sequences(sequences, n_steps_in, n_steps_out):
        #https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
    X, y, u = list(), list(),list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y, seq_u= sequences[i:end_ix, :], sequences[end_ix:out_end_ix+1, :],sequences[end_ix-1:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
        u.append(seq_u)
    return np.array(X), np.array(y), np.array(u)# choose a number of time steps #change this accordingly