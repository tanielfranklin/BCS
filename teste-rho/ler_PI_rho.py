import matplotlib.pyplot as plt
import numpy as np

def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False

def ler_dados_rho_PI(str):
    with open(str, 'r') as f:
        d = f.readlines()
        epocas = np.zeros(len(d));
        param = np.zeros((len(d),2));
        j = 0;
        data = []
        for i in d:
            k = i.rstrip().split(" [") # cada espaรงo divide a linha em duas colunas 
            data.append([float(i) if is_float(i) else i for i in k])
            epocas[j] = float(k[0])
            #print('k',k)
            #n = n.rstrip("]")
            k[1]=k[1].rstrip("]")
            aux=k[1].rstrip().split(", ")
            param[j,:]=np.array([float(aux[0]),float(aux[1])])
            #tau[j]=float(k[1])
            j += 1;
    return param, epocas

def plot_rho_PI(str_file):
    param,epocas=ler_dados_rho_PI(str_file)
    label=[r"$\rho$", 'PI']
    print(param.shape)
    fig=plt.figure()
    ax=fig.add_subplot()
    ln1=ax.plot(epocas,param[:,0],'red', label=label[0])
    ax2=ax.twinx()
    ax2.set_ylabel('PI (1e-9)')
    ln2=ax2.plot(epocas,param[:,1]*1e9,'y', label=label[1])
    # added these three lines
    ln = ln1+ln2
    labs = [l.get_label() for l in ln]
    ax2.legend(ln, labs, loc=0)
    ax.set_ylabel(r"$\rho$")
    ax.set_xlabel('Epocas')
    plt.grid(True)
    plt.show()


plot_rho_PI('rho_PI.dat')